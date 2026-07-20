export type CoordinatorMode = "local" | "remote";
export type AdapterMode = "connectivity-test" | "local-model-runtime";

export interface DesktopSettings {
  coordinatorMode: CoordinatorMode;
  remoteCoordinatorUrl: string;
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
  status: "online" | "suspect" | "offline" | "draining";
  connected: boolean;
  region: string;
  offeredVramMb: number;
  reliability: number;
  jobsCompleted: number;
  lastSeenAt: string;
  gpus: DashboardGpu[];
  deployments: DashboardDeployment[];
}

export interface DashboardModel {
  id: string;
  replicas: number;
  pipelines: number;
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
  localHardware: LocalHardware;
  settings: DesktopSettings;
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
  totalTokens: number;
  routeClass: string;
}

export interface DesktopBridge {
  getSnapshot(): Promise<DashboardSnapshot>;
  saveSettings(settings: DesktopSettings): Promise<DashboardSnapshot>;
  setContribution(enabled: boolean): Promise<DashboardSnapshot>;
  sendChat(request: ChatRequest): Promise<ChatResponse>;
}
