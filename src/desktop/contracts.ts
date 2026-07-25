export type CoordinatorMode = "local" | "remote";
import type {
  ChatMessage,
  ComputeMode,
  HubCatalogModel,
  HubCatalogPage,
  HubCatalogSearchInput,
  HubCatalogSort,
  WorkerAcceleratorDiagnostics,
} from "../contracts/types.js";
export type { ComputeMode } from "../contracts/types.js";
export type { HubCatalogModel, HubCatalogPage, HubCatalogSearchInput, HubCatalogSort } from "../contracts/types.js";

export interface DesktopSettings {
  coordinatorMode: CoordinatorMode;
  remoteCoordinatorUrl: string;
  remoteCoordinatorToken: string;
  contributionEnabled: boolean;
  computeMode: ComputeMode;
  launchAtLogin: boolean;
  closeToTray: boolean;
  onboardingComplete: boolean;
  region: string;
  offeredVramMb: number;
}

export interface DashboardGpu {
  id: string;
  vendor: string;
  model: string;
  physicalVramMb: number;
  sharedMemoryMb?: number;
  offeredVramMb: number;
  freeOfferedVramMb: number;
  utilizationPct?: number;
  temperatureC?: number;
  powerW?: number;
}

export interface DashboardDeployment {
  deploymentId: string;
  model: string;
  modelDigest: string;
  mode: "replica" | "pipeline";
  adapter: string;
  peakVramMb: number;
  contextLimit: number;
  freeSlots: number;
  tokensPerSecond: number;
  throughputSource?: "measured" | "estimated" | "configured" | "default";
  ttftMs: number;
  execution?: {
    deviceType: "cpu" | "gpu" | "mixed";
    backend: "cpu" | "cuda" | "rocm" | "directml" | "mps" | "vulkan" | "webgpu";
    deviceName: string;
    precision: string;
    fallback: boolean;
    fallbackReason?: string;
    stages?: Array<{
      nodeId: string;
      stageIndex: number;
      layerStart: number;
      layerEnd: number;
      deviceType: "cpu" | "gpu";
      backend: "cpu" | "cuda" | "rocm" | "directml" | "mps" | "vulkan" | "webgpu";
      deviceName: string;
      precision: string;
      fallback: boolean;
      fallbackReason?: string;
    }>;
  };
}

export interface DashboardWorker {
  id: string;
  kind: "desktop" | "browser" | "cell";
  status: "online" | "suspect" | "offline" | "draining";
  connected: boolean;
  region: string;
  offeredVramMb: number;
  reliability: number;
  jobsCompleted: number;
  lastSeenAt: string;
  gpus: DashboardGpu[];
  deployments: DashboardDeployment[];
  /** Stable runtime node identity used to bind effective stage telemetry. */
  executionNodeId?: string;
  computeMode?: ComputeMode;
  agentVersion?: string;
  acceleration?: WorkerAcceleratorDiagnostics;
  mobile?: {
    platform: string;
    backend: "webgpu" | "cpu";
    performanceLevel: "low" | "balanced" | "maximum";
    wakeLock: boolean;
    estimatedGflops: number;
    verifiedTasks: number;
    residentExperts: Array<{
      artifactId: string;
      modelId: string;
      modelDigest: string;
      layer: number;
      expert: number;
      contentId: string;
      bytes: number;
    }>;
  };
}

export interface DashboardModel {
  id: string;
  replicas: number;
  pipelines: number;
}

export interface RequestedModelCapacity {
  id: string;
  source: string;
  revision: string | null;
  status: "profiling" | "waiting_capacity" | "ready" | "activating" | "active" | "incompatible" | "failed";
  autoActivate: boolean;
  adapterId: string | null;
  compatible: boolean | null;
  requiredVramMiB: number | null;
  availableVramMiB: number;
  missingVramMiB: number | null;
  requiredNodes: number;
  availableNodes: number;
  missingNodes: number;
  weightBytes: number | null;
  contextTokens: number;
  message: string;
  activationProgress: Array<{
    phase: string;
    message: string;
    at: string;
    state: "running" | "completed" | "failed";
    nodeId?: string;
    processId?: string;
    device?: string;
    details?: string[];
  }>;
  activationRequestedAt: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface RequestModelInput {
  id: string;
  source: string;
  revision: string | null;
  contextTokens: number;
  minimumNodes: number;
  autoActivate: boolean;
}

export interface DashboardJob {
  id: string;
  model: string;
  status: string;
  workerId: string | null;
  inputTokens: number;
  outputTokens: number;
  failureCode: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface LocalHardware {
  hostname: string;
  platform: string;
  ramMb: number;
  gpus: Array<{
    id: string;
    vendor: string;
    model: string;
    physicalVramMb: number;
    sharedMemoryMb?: number | undefined;
    utilizationPct?: number | undefined;
    temperatureC?: number | undefined;
    powerW?: number | undefined;
  }>;
}

export type DesktopUpdateState =
  | "unsupported"
  | "development"
  | "idle"
  | "checking"
  | "downloading"
  | "ready"
  | "up-to-date"
  | "error";

export interface DesktopUpdateStatus {
  state: DesktopUpdateState;
  currentVersion: string;
  availableVersion: string | null;
  message: string;
  checkedAt: string | null;
}

export type AcceleratorPreparationPhase =
  | "idle"
  | "detecting"
  | "checking-cache"
  | "checking-prerequisites"
  | "copying-base"
  | "downloading"
  | "verifying-package"
  | "installing"
  | "physical-probe"
  | "activating"
  | "ready"
  | "fallback"
  | "blocked"
  | "error";

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

export interface DesktopAccelerationStatus {
  state: "idle" | "preparing" | "cpu-ready" | "gpu-ready" | "gpu-fallback" | "error";
  requestedBackend: "cpu" | "cuda" | "rocm" | "mps" | "xpu" | null;
  effectiveBackend: "cpu" | "cuda" | "rocm" | "mps" | "xpu" | null;
  deviceName: string | null;
  precision: "float32" | "float16" | null;
  message: string;
  cpu: {
    state: "unavailable" | "ready" | "active";
    activeStages: number;
    deviceName: string;
    precision: "float32";
    message: string;
  };
  gpu: {
    state:
      | "not-detected"
      | "unsupported"
      | "checking"
      | "downloading"
      | "installing"
      | "verifying"
      | "ready"
      | "fallback"
      | "error";
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

export interface DashboardSnapshot {
  capturedAt: string;
  coordinatorUrl: string;
  appVersion: string;
  platform: string;
  connectionError: string | null;
  runtimeError: string | null;
  health: {
    status: string;
    version: string;
    workers: { registered: number; connected: number; online: number };
  } | null;
  workers: DashboardWorker[];
  models: DashboardModel[];
  requestedModels: RequestedModelCapacity[];
  jobs: DashboardJob[];
  localHardware: LocalHardware;
  contribution: {
    state: "paused" | "connecting" | "connected" | "error";
    workerId: string | null;
  };
  acceleration: DesktopAccelerationStatus;
  modelAdminAuthorization: {
    configured: boolean;
    encrypted: boolean;
  };
  settings: DesktopSettings;
  update: DesktopUpdateStatus;
}

export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  sessionId: string;
  maxTokens?: number;
}

export interface ChatResponse {
  requestId: string;
  model: string;
  text: string;
  promptTokens: number;
  outputTokens: number;
  totalTokens: number;
  routeClass: string;
  affinityHit: boolean;
  sessionId: string;
  reusedKvTokens: number;
  ttftMs: number;
  activeMs: number;
}

export interface ChatStreamUpdate {
  requestId: string;
  model: string;
  delta: string;
  text: string;
  outputTokens: number;
  routeClass: string;
  affinityHit: boolean;
  sessionId: string;
  reusedKvTokens: number;
  ttftMs: number;
  elapsedMs: number;
  phase?: "connecting" | "waiting_first_token" | "recovering" | "streaming";
  statusMessage?: string;
  attempt?: number;
  maximumAttempts?: number;
  affectedWorkerId?: string;
  affectedNodeId?: string;
}

export interface SupportAssistantPublicConfig {
  enabled: boolean;
  available: boolean;
  provider: "mycellios-network";
  configuredModel: string | null;
  selectedModel: string | null;
  availableModels: string[];
  welcomeMessage: string;
  suggestions: string[];
  allowDeviceControl: boolean;
  updatedAt: string | null;
}

export interface SupportAssistantAdminSettings {
  enabled: boolean;
  modelId: string | null;
  systemPrompt: string;
  welcomeMessage: string;
  suggestions: string[];
  maxOutputTokens: number;
  temperature: number;
  allowDeviceControl: boolean;
  updatedAt: number;
}

export interface SupportAssistantAdminResponse {
  settings: SupportAssistantAdminSettings;
  runtime: SupportAssistantPublicConfig;
}

export interface SupportAssistantChatRequest {
  sessionId: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  page?: string;
  platform?: string;
}

export interface DesktopBridge {
  getSnapshot(): Promise<DashboardSnapshot>;
  saveSettings(settings: DesktopSettings): Promise<DashboardSnapshot>;
  setContribution(enabled: boolean): Promise<DashboardSnapshot>;
  sendChat(request: ChatRequest): Promise<ChatResponse>;
  streamChat?(request: ChatRequest, onUpdate: (update: ChatStreamUpdate) => void): Promise<ChatResponse>;
  getSupportAssistantConfig(): Promise<SupportAssistantPublicConfig>;
  streamSupportAssistant(
    request: SupportAssistantChatRequest,
    onUpdate: (update: ChatStreamUpdate) => void,
  ): Promise<ChatResponse>;
  getSupportAssistantAdmin(adminToken?: string): Promise<SupportAssistantAdminResponse>;
  saveSupportAssistantAdmin(
    settings: Omit<SupportAssistantAdminSettings, "updatedAt">,
    adminToken?: string,
  ): Promise<SupportAssistantAdminResponse>;
  removeWorker(workerId: string): Promise<DashboardSnapshot>;
  clearOfflineWorkers(): Promise<DashboardSnapshot>;
  searchHubModels(input: HubCatalogSearchInput): Promise<HubCatalogPage>;
  requestModel(input: RequestModelInput, adminToken?: string): Promise<DashboardSnapshot>;
  removeRequestedModel(modelId: string, adminToken?: string): Promise<DashboardSnapshot>;
  getBenchmarkRuns(): Promise<BenchmarkRun[]>;
  runBenchmark(): Promise<BenchmarkRun>;
  checkForUpdates(): Promise<DesktopUpdateStatus>;
  installUpdate(): Promise<void>;
  minimizeWindow(): Promise<void>;
  toggleMaximizeWindow(): Promise<boolean>;
  closeWindow(): Promise<void>;
}
import type { BenchmarkRun } from "../benchlab/types.js";
