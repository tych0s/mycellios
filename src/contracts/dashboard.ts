import type {
  ChatMessage,
  ComputeMode,
  HubCatalogModel,
  HubCatalogPage,
  HubCatalogSearchInput,
  HubCatalogSort,
  WorkerAcceleratorDiagnostics,
  WorkerExecutorIsolationCapability,
} from "../contracts/types.js";
import type {
  ComponentUpdateChannel,
  CoordinatorMode,
  DeveloperChannelEnrollment,
  NodeControlSettings,
} from "../contracts/node-settings.js";
import type { NodeAccelerationStatus } from "../contracts/acceleration.js";
import type { ChatRequest, ChatResponse, ChatStreamUpdate } from "../contracts/chat.js";
import type {
  SystemLogEntry,
  SystemLogLevel,
  SystemLogSnapshot,
} from "../contracts/system-log.js";
import type { NativeBuildIdentity } from "../contracts/build-identity.js";
import type { ActivationIncident } from "./activation-incident.js";
import type { ChatActivity } from "../contracts/chat-activity.js";
import type {
  FleetContributionCommandResponse,
  FleetContributionStatus,
} from "../contracts/fleet-contribution.js";
export type { ComputeMode } from "../contracts/types.js";
export type { HubCatalogModel, HubCatalogPage, HubCatalogSearchInput, HubCatalogSort } from "../contracts/types.js";
export type {
  FleetContributionCommandResponse,
  FleetContributionStatus,
} from "../contracts/fleet-contribution.js";
export type { ComponentUpdateChannel, CoordinatorMode } from "../contracts/node-settings.js";
export type DesktopSettings = NodeControlSettings;
export type DesktopDeveloperChannelEnrollment = DeveloperChannelEnrollment;
export type DesktopAccelerationStatus = NodeAccelerationStatus;
export type {
  AcceleratorPreparationIssue,
  AcceleratorPreparationLogEntry,
  AcceleratorPreparationPhase,
} from "../contracts/acceleration.js";
export type { SystemLogEntry, SystemLogLevel, SystemLogSnapshot } from "../contracts/system-log.js";
export type { ChatRequest, ChatResponse, ChatStreamUpdate } from "../contracts/chat.js";

export interface DesktopDevelopmentLabInvitation {
  labId: string;
  feedUrl: string;
  expiresAt: string;
}

export interface DesktopDevelopmentLabCreation {
  snapshot: DashboardSnapshot;
  invitation: DesktopDevelopmentLabInvitation;
}

export type ComponentUpdateState =
  | "disabled"
  | "idle"
  | "checking"
  | "downloading"
  | "waiting-idle"
  | "canarying"
  | "updating"
  | "activating"
  | "rollback"
  | "up-to-date"
  | "error";

export interface DesktopComponentUpdateStatus {
  channel: ComponentUpdateChannel;
  state: ComponentUpdateState;
  activeSequence: number | null;
  availableSequence: number | null;
  activeRevision: string | null;
  availableRevision: string | null;
  message: string;
  checkedAt: string | null;
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
  buildIdentity?: NativeBuildIdentity;
  acceleration?: WorkerAcceleratorDiagnostics;
  isolation?: WorkerExecutorIsolationCapability;
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
  activationIncident: ActivationIncident | null;
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
  components?: DesktopComponentUpdateStatus | undefined;
}

export type DeveloperChannelState =
  | "stopped"
  | "starting"
  | "ready"
  | "stopping"
  | "error";

export type DeveloperOperationState =
  | "idle"
  | "running"
  | "cancelling"
  | "passed"
  | "failed"
  | "cancelled";

export interface DesktopDeveloperWorkflowStatus {
  available: boolean;
  unavailableReason: string | null;
  channel: {
    state: DeveloperChannelState;
    feedUrl: string | null;
    keyId: string | null;
    fingerprint: string | null;
    enrolled: boolean;
    message: string;
  };
  publication: {
    state: DeveloperOperationState;
    message: string;
    revision: string | null;
    completedAt: string | null;
  };
  distributedGate: {
    state: DeveloperOperationState;
    phase: string | null;
    message: string;
    startedAt: string | null;
    completedAt: string | null;
    log: string[];
  };
}

export interface DashboardSnapshot {
  capturedAt: string;
  coordinatorUrl: string;
  appVersion: string;
  buildIdentity: NativeBuildIdentity | null;
  platform: string;
  connectionError: string | null;
  runtimeError: string | null;
  health: {
    status: string;
    version: string;
    buildIdentity: NativeBuildIdentity | null;
    workers: { registered: number; connected: number; online: number };
  } | null;
  workers: DashboardWorker[];
  models: DashboardModel[];
  requestedModels: RequestedModelCapacity[];
  jobs: DashboardJob[];
  recentNetworkTraces?: import("../contracts/types.js").NetworkExecutionTrace[];
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
  developerWorkflow: DesktopDeveloperWorkflowStatus;
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

export interface WorkerCredentialSummary {
  identityKind: "device" | "cell" | "browser";
  identityId: string;
  fingerprint: string;
  status: "active" | "revoked";
  protocolVersion: number;
  createdAt: string;
  updatedAt: string;
  lastSeenAt: string;
  revokedAt: string | null;
  revocationReason: string | null;
}

export interface WorkerCredentialRevocationResponse {
  state: "revoked" | "already_revoked";
  disconnected: number;
  credential: WorkerCredentialSummary;
}

export interface SupportAssistantChatRequest {
  sessionId: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  page?: string;
  platform?: string;
}

export interface DesktopBridge {
  getSnapshot(): Promise<DashboardSnapshot>;
  getSystemLogs(): Promise<SystemLogSnapshot>;
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
  getFleetContributionAdmin(adminToken?: string): Promise<FleetContributionStatus>;
  setFleetContributionAdmin(
    enabled: boolean,
    adminToken?: string,
  ): Promise<FleetContributionCommandResponse>;
  removeWorker(workerId: string): Promise<DashboardSnapshot>;
  clearOfflineWorkers(): Promise<DashboardSnapshot>;
  listWorkerCredentials(adminToken?: string): Promise<WorkerCredentialSummary[]>;
  revokeWorkerCredential(
    credential: Pick<WorkerCredentialSummary, "identityKind" | "identityId" | "fingerprint">,
    reason: string,
    adminToken?: string,
  ): Promise<WorkerCredentialRevocationResponse>;
  searchHubModels(input: HubCatalogSearchInput): Promise<HubCatalogPage>;
  requestModel(input: RequestModelInput, adminToken?: string): Promise<DashboardSnapshot>;
  removeRequestedModel(modelId: string, adminToken?: string): Promise<DashboardSnapshot>;
  getBenchmarkRuns(): Promise<BenchmarkRun[]>;
  runBenchmark(): Promise<BenchmarkRun>;
  checkForUpdates(): Promise<DesktopUpdateStatus>;
  installUpdate(): Promise<void>;
  enrollDeveloperComponentChannel(
    enrollment: DesktopDeveloperChannelEnrollment,
  ): Promise<DashboardSnapshot>;
  leaveDeveloperComponentChannel(): Promise<DashboardSnapshot>;
  createDeveloperRemoteLab(): Promise<DesktopDevelopmentLabCreation>;
  joinDeveloperRemoteLab(): Promise<DashboardSnapshot>;
  revokeDeveloperRemoteLab(): Promise<DashboardSnapshot>;
  initializeDeveloperWorkflow(): Promise<DashboardSnapshot>;
  stopDeveloperWorkflow(): Promise<DashboardSnapshot>;
  publishDeveloperRuntime(): Promise<DesktopDeveloperWorkflowStatus>;
  cancelDeveloperRuntimePublication(): Promise<DesktopDeveloperWorkflowStatus>;
  runDeveloperDistributedGate(): Promise<DesktopDeveloperWorkflowStatus>;
  cancelDeveloperDistributedGate(): Promise<DesktopDeveloperWorkflowStatus>;
  minimizeWindow(): Promise<void>;
  toggleMaximizeWindow(): Promise<boolean>;
  closeWindow(): Promise<void>;
}
import type { BenchmarkRun } from "../benchlab/types.js";
