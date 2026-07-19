export type WorkloadClass = "interactive" | "batch" | "benchmark";
export type AdapterKind = "mock" | "local-model-runtime" | "externalggufruntime" | "openai-compatible";
export type ExecutionMode = "replica" | "pipeline";
export type WorkerStatus = "online" | "suspect" | "offline" | "draining";
export type JobStatus =
  | "queued"
  | "leasing"
  | "running"
  | "streaming"
  | "completed"
  | "failed"
  | "cancelled"
  | "expired";

export interface ChatMessage {
  role: "system" | "developer" | "user" | "assistant" | "tool";
  content: string;
  name?: string;
}

export interface ChatCompletionRequest {
  model: string;
  messages: ChatMessage[];
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  seed?: number;
  session_id?: string;
  workload_class?: WorkloadClass;
  preferred_region?: string;
  deadline_ms?: number;
}

export interface ModelDeployment {
  deploymentId: string;
  model: string;
  modelDigest: string;
  mode: ExecutionMode;
  adapter: AdapterKind;
  peakVramMb: number;
  contextLimit: number;
  maxConcurrency: number;
  freeSlots: number;
  tokensPerSecond: number;
  ttftMs: number;
  dataLocality: "local" | "external";
  stage?: {
    index: number;
    total: number;
    layerStart: number;
    layerEnd: number;
  } | undefined;
}

export interface GpuCapability {
  id: string;
  vendor: string;
  model: string;
  physicalVramMb: number;
  sharedMemoryMb?: number | undefined;
  unifiedMemory?: boolean | undefined;
  offeredVramMb: number;
  freeOfferedVramMb: number;
  utilizationPct?: number | undefined;
  temperatureC?: number | undefined;
  powerW?: number | undefined;
}

export interface WorkerLimits {
  maxConcurrency: number;
  maxPowerW?: number | undefined;
  maxTemperatureC?: number | undefined;
  pauseWhenForeground: boolean;
}

export interface LlmfitGpuAdvisory {
  name: string;
  backend: string;
  vramMb: number;
  unifiedMemory: boolean;
}

export interface LlmfitModelAdvisory {
  deploymentId?: string | undefined;
  requestedModel: string;
  resolvedModel: string;
  fitLevel: string;
  runMode: string;
  runtime?: string | undefined;
  bestQuant?: string | undefined;
  estimatedTokensPerSecond?: number | undefined;
  measuredTokensPerSecond?: number | undefined;
  memoryRequiredMb?: number | undefined;
  usableContext?: number | undefined;
}

/**
 * Node-local advice from llmfit. This is intentionally separate from the
 * canonical deployment and pipeline profiles: whole-model fit does not prove
 * that a node is suitable (or unsuitable) for an individual GDLP stage.
 */
export interface LlmfitAdvisory {
  source: "llmfit";
  scope: "host";
  backend: string;
  cpuName: string;
  cpuCores: number;
  totalRamMb: number;
  availableRamMb: number;
  gpuCount: number;
  gpus: LlmfitGpuAdvisory[];
  model?: LlmfitModelAdvisory | undefined;
}

export interface WorkerCapabilities {
  region: string;
  agentVersion: string;
  gpus: GpuCapability[];
  limits: WorkerLimits;
  deployments: ModelDeployment[];
  network: {
    coordinatorRttMs: number;
    uplinkMbps: number;
    downlinkMbps: number;
  };
  llmfit?: LlmfitAdvisory | undefined;
}

export interface WorkerRegistration {
  capabilities: WorkerCapabilities;
}

export interface WorkerHeartbeat {
  draining: boolean;
  pausedReason?: string | null;
  activeLeases: string[];
  gpus: Array<
    Pick<
      GpuCapability,
      "id" | "freeOfferedVramMb" | "utilizationPct" | "temperatureC" | "powerW"
    >
  >;
  deployments: Array<Pick<ModelDeployment, "deploymentId" | "freeSlots">>;
  network: Pick<WorkerCapabilities["network"], "coordinatorRttMs" | "uplinkMbps">;
}

export interface CompletionMetrics {
  inputTokens: number;
  outputTokens: number;
  ttftMs: number;
  activeMs: number;
  energyWh?: number | undefined;
}

export interface WorkerEnvelope<T = unknown> {
  v: 1;
  type: string;
  workerId: string;
  payload: T;
}

export interface ServerEnvelope<T = unknown> {
  v: 1;
  type: string;
  payload: T;
}

export interface RouteStage {
  workerId: string;
  deploymentId: string;
  modelDigest: string;
  stageIndex: number;
  score: number;
}

export interface ScheduledRoute {
  routeClass: ExecutionMode;
  model: string;
  region: string;
  stages: RouteStage[];
  score: number;
  affinityHit: boolean;
}

export interface ScheduledRoutePlan {
  primary: ScheduledRoute;
  standbys: ScheduledRoute[];
}

export interface JobPayload {
  jobId: string;
  leaseId: string;
  modelDigest: string;
  deadlineAt: number;
  request: ChatCompletionRequest;
}

export interface TokenEvent {
  index: number;
  text: string;
}

export interface CompletionResult {
  jobId: string;
  leaseId: string;
  text: string;
  finishReason: "stop" | "length" | "cancelled" | "error";
  metrics: CompletionMetrics;
}
