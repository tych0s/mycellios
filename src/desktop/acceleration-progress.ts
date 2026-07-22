import type {
  AcceleratorPreparationIssue,
  AcceleratorPreparationLogEntry,
  AcceleratorPreparationPhase,
  DesktopAccelerationStatus,
} from "./contracts.js";

export const MAX_ACCELERATION_LOG_ENTRIES = 60;

const ACTIVE_GPU_PREPARATION_PHASES = new Set<AcceleratorPreparationPhase>([
  "detecting",
  "checking-cache",
  "checking-prerequisites",
  "copying-base",
  "downloading",
  "verifying-package",
  "installing",
  "physical-probe",
  "activating",
]);

export function gpuPreparationIsContinuing(status: DesktopAccelerationStatus): boolean {
  if (ACTIVE_GPU_PREPARATION_PHASES.has(status.preparation.phase)) return true;
  return Boolean(
    status.preparation.issue?.retryable
    && (status.preparation.phase === "fallback" || status.preparation.phase === "error"),
  );
}

/**
 * Return the best runtime that is already usable without ever awaiting the
 * background GPU installer. Recheck after loading CPU so a GPU that completed
 * during that short read can still win for this new launch.
 */
export async function selectImmediateRuntime<T>(
  resolvedRuntime: () => T | null,
  loadCpuRuntime: () => Promise<T>,
): Promise<T> {
  const resolved = resolvedRuntime();
  if (resolved !== null) return resolved;
  const cpu = await loadCpuRuntime();
  return resolvedRuntime() ?? cpu;
}

export interface DesktopAccelerationProgressUpdate {
  phase: AcceleratorPreparationPhase;
  message: string;
  progressPct?: number | null | undefined;
  bytesCompleted?: number | null | undefined;
  bytesTotal?: number | null | undefined;
  bytesPerSecond?: number | null | undefined;
  etaSeconds?: number | null | undefined;
  currentArtifact?: string | null | undefined;
  artifactIndex?: number | null | undefined;
  artifactCount?: number | null | undefined;
  issue?: AcceleratorPreparationIssue | null | undefined;
  level?: AcceleratorPreparationLogEntry["level"] | undefined;
  recordLog?: boolean | undefined;
  at?: string | undefined;
}

export function createInitialAccelerationStatus(): DesktopAccelerationStatus {
  return {
    state: "idle",
    requestedBackend: null,
    effectiveBackend: "cpu",
    deviceName: null,
    precision: "float32",
    message: "The certified CPU runtime will be available when contribution starts.",
    cpu: {
      state: "unavailable",
      activeStages: 0,
      deviceName: "CPU",
      precision: "float32",
      message: "Waiting for contribution to start.",
    },
    gpu: {
      state: "not-detected",
      vendor: null,
      model: null,
      backend: null,
      activeStages: 0,
    },
    preparation: {
      phase: "idle",
      progressPct: null,
      bytesCompleted: null,
      bytesTotal: null,
      bytesPerSecond: null,
      etaSeconds: null,
      startedAt: null,
      updatedAt: null,
      currentArtifact: null,
      artifactIndex: null,
      artifactCount: null,
      issue: null,
      log: [],
    },
  };
}

export function beginAccelerationPreparation(
  current: DesktopAccelerationStatus,
  target: {
    vendor: string | null;
    model: string | null;
    backend: DesktopAccelerationStatus["gpu"]["backend"];
    cpuName?: string | undefined;
  },
  at = new Date().toISOString(),
): DesktopAccelerationStatus {
  const backend = target.backend;
  const supported = backend !== null;
  const message = backend !== null
    ? `Detected ${target.model ?? "GPU"}; preparing the ${backend.toUpperCase()} runtime in the background.`
    : target.model
      ? `${target.model} has no certified native GPU pack in this release; CPU remains available.`
      : "No supported GPU was detected; CPU remains available.";
  const next: DesktopAccelerationStatus = {
    ...current,
    state: supported ? "preparing" : "cpu-ready",
    requestedBackend: target.backend ?? "cpu",
    effectiveBackend: "cpu",
    deviceName: target.model,
    precision: "float32",
    message,
    cpu: {
      ...current.cpu,
      state: current.cpu.activeStages > 0 ? "active" : "ready",
      deviceName: target.cpuName?.trim() || current.cpu.deviceName,
      message: current.cpu.activeStages > 0
        ? `${current.cpu.activeStages} model stage${current.cpu.activeStages === 1 ? " is" : "s are"} running on CPU.`
        : "Certified CPU runtime ready while GPU setup continues.",
    },
    gpu: {
      ...current.gpu,
      state: supported ? "checking" : target.model ? "unsupported" : "not-detected",
      vendor: target.vendor,
      model: target.model,
      backend: target.backend,
    },
    preparation: {
      phase: supported ? "detecting" : "fallback",
      progressPct: supported ? 0 : null,
      bytesCompleted: null,
      bytesTotal: null,
      bytesPerSecond: null,
      etaSeconds: null,
      startedAt: at,
      updatedAt: at,
      currentArtifact: null,
      artifactIndex: null,
      artifactCount: null,
      issue: null,
      log: current.preparation.log,
    },
  };
  return appendAccelerationLog(next, {
    at,
    level: supported ? "info" : "warning",
    message,
  });
}

export function applyAccelerationProgress(
  current: DesktopAccelerationStatus,
  update: DesktopAccelerationProgressUpdate,
): DesktopAccelerationStatus {
  const at = update.at ?? new Date().toISOString();
  const progressPct = normalizeProgress(current.preparation.progressPct, update.progressPct, update.phase);
  const issueCode = update.issue?.code ?? current.preparation.issue?.code ?? null;
  const terminalState = update.phase === "ready"
    ? "gpu-ready"
    : update.phase === "fallback" || update.phase === "blocked"
      ? "gpu-fallback"
      : update.phase === "error"
        ? "error"
        : "preparing";
  const gpuState: DesktopAccelerationStatus["gpu"]["state"] =
    update.phase === "downloading"
      ? "downloading"
      : update.phase === "installing"
        ? "installing"
          : update.phase === "verifying-package" || update.phase === "physical-probe" || update.phase === "activating"
          ? "verifying"
          : update.phase === "ready"
            ? "ready"
            : update.phase === "fallback" || update.phase === "blocked"
              ? terminalGpuState(current.gpu.state, issueCode)
              : update.phase === "error"
                ? "error"
                : "checking";
  const effectiveBackend = update.phase === "ready"
    ? current.gpu.backend
    : "cpu";
  let next: DesktopAccelerationStatus = {
    ...current,
    state: terminalState,
    effectiveBackend,
    precision: update.phase === "ready" ? "float16" : "float32",
    message: update.message,
    cpu: {
      ...current.cpu,
      state: current.cpu.activeStages > 0 ? "active" : "ready",
      message: current.cpu.activeStages > 0
        ? `${current.cpu.activeStages} model stage${current.cpu.activeStages === 1 ? " is" : "s are"} running on CPU.`
        : update.phase === "ready"
          ? "Certified CPU fallback remains ready."
          : "Certified CPU runtime ready while GPU setup continues.",
    },
    gpu: {
      ...current.gpu,
      state: gpuState,
    },
    preparation: {
      ...current.preparation,
      phase: update.phase,
      progressPct,
      bytesCompleted: provided(update.bytesCompleted, current.preparation.bytesCompleted),
      bytesTotal: provided(update.bytesTotal, current.preparation.bytesTotal),
      bytesPerSecond: provided(update.bytesPerSecond, current.preparation.bytesPerSecond),
      etaSeconds: provided(update.etaSeconds, current.preparation.etaSeconds),
      startedAt: current.preparation.startedAt ?? at,
      updatedAt: at,
      currentArtifact: provided(update.currentArtifact, current.preparation.currentArtifact),
      artifactIndex: provided(update.artifactIndex, current.preparation.artifactIndex),
      artifactCount: provided(update.artifactCount, current.preparation.artifactCount),
      issue: update.issue === undefined ? current.preparation.issue : update.issue,
    },
  };
  if (update.recordLog !== false) {
    next = appendAccelerationLog(next, {
      at,
      level: update.level ?? phaseLevel(update.phase),
      message: update.message,
    });
  }
  return next;
}

export function updateActiveAccelerationStages(
  current: DesktopAccelerationStatus,
  deviceType: "cpu" | "gpu",
  delta: 1 | -1,
  message?: string,
  at = new Date().toISOString(),
): DesktopAccelerationStatus {
  const cpuStages = deviceType === "cpu"
    ? Math.max(0, current.cpu.activeStages + delta)
    : current.cpu.activeStages;
  const gpuStages = deviceType === "gpu"
    ? Math.max(0, current.gpu.activeStages + delta)
    : current.gpu.activeStages;
  let next: DesktopAccelerationStatus = {
    ...current,
    cpu: {
      ...current.cpu,
      activeStages: cpuStages,
      state: cpuStages > 0 ? "active" : "ready",
      message: cpuStages > 0
        ? `${cpuStages} model stage${cpuStages === 1 ? " is" : "s are"} running on CPU.`
        : current.state === "preparing"
          ? "Certified CPU runtime ready while GPU setup continues."
          : "Certified CPU fallback ready.",
    },
    gpu: {
      ...current.gpu,
      activeStages: gpuStages,
    },
  };
  if (message) {
    next = appendAccelerationLog(next, {
      at,
      level: delta > 0 ? "success" : "info",
      message,
    });
  }
  return next;
}

export function appendAccelerationLog(
  current: DesktopAccelerationStatus,
  entry: AcceleratorPreparationLogEntry,
): DesktopAccelerationStatus {
  const clean = sanitizeLogMessage(entry.message);
  if (!clean) return current;
  const previous = current.preparation.log.at(-1);
  if (previous?.message === clean && previous.level === entry.level) return current;
  const log = [...current.preparation.log, { ...entry, message: clean }]
    .slice(-MAX_ACCELERATION_LOG_ENTRIES);
  return {
    ...current,
    preparation: {
      ...current.preparation,
      log,
      updatedAt: entry.at,
    },
  };
}

function normalizeProgress(
  current: number | null,
  candidate: number | null | undefined,
  phase: AcceleratorPreparationPhase,
): number | null {
  if (phase === "ready") return 100;
  if (candidate === undefined || candidate === null || !Number.isFinite(candidate)) return current;
  const bounded = Math.max(0, Math.min(100, Math.round(candidate * 10) / 10));
  return current === null ? bounded : Math.max(current, bounded);
}

function terminalGpuState(
  current: DesktopAccelerationStatus["gpu"]["state"],
  issueCode: string | null,
): DesktopAccelerationStatus["gpu"]["state"] {
  if (issueCode === "no-gpu") return "not-detected";
  if (issueCode?.startsWith("unsupported-")) return "unsupported";
  if (current === "not-detected" || current === "unsupported") return current;
  return "fallback";
}

function provided<T>(candidate: T | undefined, current: T): T {
  return candidate === undefined ? current : candidate;
}

function phaseLevel(phase: AcceleratorPreparationPhase): AcceleratorPreparationLogEntry["level"] {
  if (phase === "ready") return "success";
  if (phase === "fallback" || phase === "blocked") return "warning";
  if (phase === "error") return "error";
  return "info";
}

function sanitizeLogMessage(message: string): string {
  return message
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\bBearer\s+\S+/gi, "Bearer [redacted]")
    .replace(/([?&](?:access_?token|api_?key|secret|token)=)[^&\s]+/gi, "$1[redacted]")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 600);
}
