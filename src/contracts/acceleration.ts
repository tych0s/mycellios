export type AcceleratorPreparationPhase =
  | "idle" | "detecting" | "checking-cache" | "checking-prerequisites"
  | "copying-base" | "downloading" | "verifying-package" | "installing"
  | "physical-probe" | "activating" | "ready" | "fallback" | "blocked" | "error";

export interface AcceleratorPreparationIssue {
  code: string;
  message: string;
  action: string;
  retryable: boolean;
  requiredBytes?: number | undefined;
  availableBytes?: number | undefined;
}

export interface AcceleratorPreparationLogEntry {
  at: string;
  level: "info" | "success" | "warning" | "error";
  message: string;
}

export interface NodeAccelerationStatus {
  state: "idle" | "preparing" | "cpu-ready" | "gpu-ready" | "gpu-fallback" | "error";
  requestedBackend: "cpu" | "cuda" | "rocm" | "mps" | "xpu" | null;
  effectiveBackend: "cpu" | "cuda" | "rocm" | "mps" | "xpu" | null;
  deviceName: string | null;
  precision: "float32" | "float16" | null;
  message: string;
  cpu: { state: "unavailable" | "ready" | "active"; activeStages: number; deviceName: string; precision: "float32"; message: string };
  gpu: {
    state: "not-detected" | "unsupported" | "checking" | "downloading" | "installing" | "verifying" | "ready" | "fallback" | "error";
    vendor: string | null;
    model: string | null;
    backend: "cuda" | "rocm" | "mps" | "xpu" | null;
    activeStages: number;
  };
  preparation: {
    phase: AcceleratorPreparationPhase;
    progressPct: number | null;
    bytesCompleted: number | null;
    bytesTotal: number | null;
    bytesPerSecond: number | null;
    etaSeconds: number | null;
    startedAt: string | null;
    updatedAt: string | null;
    currentArtifact: string | null;
    artifactIndex: number | null;
    artifactCount: number | null;
    issue: AcceleratorPreparationIssue | null;
    log: AcceleratorPreparationLogEntry[];
  };
}
