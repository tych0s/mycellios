import { contextBridge, ipcRenderer } from "electron";
import type {
  ChatRequest,
  DesktopBridge,
  DesktopSettings,
  RequestModelInput,
} from "./contracts.js";

const bridge: DesktopBridge = Object.freeze({
  getSnapshot: () => ipcRenderer.invoke("dashboard:read"),
  saveSettings: (settings: DesktopSettings) => ipcRenderer.invoke("settings:save", settings),
  setContribution: (enabled: boolean) => ipcRenderer.invoke("contribution:set", enabled),
  sendChat: (request: ChatRequest) => ipcRenderer.invoke("chat:send", request),
  removeWorker: (workerId: string) => ipcRenderer.invoke("workers:remove", workerId),
  clearOfflineWorkers: () => ipcRenderer.invoke("workers:clear-offline"),
  requestModel: (input: RequestModelInput) => ipcRenderer.invoke("models:request", input),
  removeRequestedModel: (modelId: string) => ipcRenderer.invoke("models:remove-request", modelId),
  getBenchmarkRuns: () => ipcRenderer.invoke("benchmarks:read"),
  runBenchmark: () => ipcRenderer.invoke("benchmarks:run"),
  checkForUpdates: () => ipcRenderer.invoke("updates:check"),
  installUpdate: () => ipcRenderer.invoke("updates:install"),
  minimizeWindow: () => ipcRenderer.invoke("window:minimize"),
  toggleMaximizeWindow: () => ipcRenderer.invoke("window:toggle-maximize"),
  closeWindow: () => ipcRenderer.invoke("window:close"),
});

contextBridge.exposeInMainWorld("mycellios", bridge);
