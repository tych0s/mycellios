export type WorkloadClass = "interactive" | "batch" | "benchmark";
export type AdapterKind = "mock" | "local-model-runtime" | "externalggufruntime" | "openai-compatible";
export type ExecutionMode = "replica" | "pipeline";
export type ExecutionDeviceType = "cpu" | "gpu" | "mixed";
export type ExecutionBackend =
  | "cpu"
  | "cuda"
  | "rocm"
  | "directml"
  | "mps"
  | "xpu"
  | "vulkan"
  | "webgpu";
export type WorkerStatus = "online" | "suspect" | "offline" | "draining";
export type WorkerIdentityKind = "device" | "cell";
export type ComputeMode = "automatic" | "gpu-only" | "cpu-only";
export type JobStatus =
  | "queued"
  | "leasing"
  | "running"
  | "streaming"
  | "completed"
  | "failed"
  | "cancelled"
  | "expired";

export interface HubCatalogModel {
  id: string;
  author: string;
  downloads: number;
  likes: number;
  lastModified: string | null;
  pipelineTag: string | null;
  modelType: string | null;
  architecture: string | null;
  adapterId: string | null;
  compatible: boolean;
  gated: boolean;
  compatibilityReason: string | null;
  parameterCount?: number | null;
  estimatedMemoryMiB?: number | null;
  memoryEstimateSource?: "hub_metadata" | "model_name" | null;
}

export type HubCatalogSort = "downloads" | "likes" | "lastModified";

export interface HubCatalogSearchInput {
  query: string;
  cursor?: string | null;
  sort?: HubCatalogSort;
  limit?: number;
}

export interface HubCatalogPage {
  data: HubCatalogModel[];
  nextCursor: string | null;
}

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
  internalPipeline?: {
    stageCount: number;
    boundaries: number[];
  } | undefined;
  /**
   * Device telemetry reported by the runtime after model weights have loaded.
   * This is deliberately separate from advertised hardware: a detected GPU is
   * not proof that inference is actually using it.
   */
  execution?: {
    deviceType: ExecutionDeviceType;
    backend: ExecutionBackend;
    deviceName: string;
    precision: string;
    fallback: boolean;
    fallbackReason?: string | undefined;
    stages?: Array<{
      nodeId: string;
      stageIndex: number;
      layerStart: number;
      layerEnd: number;
      deviceType: Exclude<ExecutionDeviceType, "mixed">;
      backend: ExecutionBackend;
      deviceName: string;
      precision: string;
      fallback: boolean;
      fallbackReason?: string | undefined;
    }> | undefined;
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
  /** A node-local shard executor controlled through the existing worker tunnel. */
  distributedExecutor?: {
    protocol: "gdlp-worker-tunnel/1" | "gdlp-worker-tunnel/2";
    nodeId: string;
    stageHost: string;
    stagePort: number;
    runtime: "python-safetensors";
    /** User-selected runtime policy. Optional only for legacy registrations. */
    computeMode?: ComputeMode | undefined;
    /** True only when this registration explicitly permits CPU model stages. */
    cpuEligible?: boolean | undefined;
  } | undefined;
}

export interface WorkerRegistration {
  identity?: {
    kind: WorkerIdentityKind;
    id: string;
  } | undefined;
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
