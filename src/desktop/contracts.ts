export type CoordinatorMode = "local" | "remote";
export type AdapterMode = "connectivity-test" | "local-model-runtime";
import type { HubCatalogModel } from "../contracts/types.js";
export type { HubCatalogModel } from "../contracts/types.js";

export interface DesktopSettings {
  coordinatorMode: CoordinatorMode;
  remoteCoordinatorUrl: string;
  remoteCoordinatorToken: string;
  contributionEnabled: boolean;
  launchAtLogin: boolean;
  closeToTray: boolean;
  onboardingComplete: boolean;
  region: string;
  offeredVramMb: number;
  adapterMode: AdapterMode;
  modelName: string;
  adapterBaseUrl: string;
  modelDigest: string;
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
  ttftMs: number;
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
  modelAdminAuthorization: {
    configured: boolean;
    encrypted: boolean;
  };
  settings: DesktopSettings;
  update: DesktopUpdateStatus;
}

export interface ChatRequest {
  model: string;
  prompt: string;
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
  ttftMs: number;
  elapsedMs: number;
}

export interface DesktopBridge {
  getSnapshot(): Promise<DashboardSnapshot>;
  saveSettings(settings: DesktopSettings): Promise<DashboardSnapshot>;
  setContribution(enabled: boolean): Promise<DashboardSnapshot>;
  sendChat(request: ChatRequest): Promise<ChatResponse>;
  streamChat?(request: ChatRequest, onUpdate: (update: ChatStreamUpdate) => void): Promise<ChatResponse>;
  removeWorker(workerId: string): Promise<DashboardSnapshot>;
  clearOfflineWorkers(): Promise<DashboardSnapshot>;
  searchHubModels(query: string): Promise<HubCatalogModel[]>;
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
