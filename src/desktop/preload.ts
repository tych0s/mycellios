import { contextBridge, ipcRenderer, type IpcRendererEvent } from "electron";
import type {
  ChatRequest,
  ChatResponse,
  ChatStreamUpdate,
  DesktopBridge,
  DesktopSettings,
  RequestModelInput,
} from "./contracts.js";

let chatStreamSequence = 0;

const bridge: DesktopBridge = Object.freeze({
  getSnapshot: () => ipcRenderer.invoke("dashboard:read"),
  saveSettings: (settings: DesktopSettings) => ipcRenderer.invoke("settings:save", settings),
  setContribution: (enabled: boolean) => ipcRenderer.invoke("contribution:set", enabled),
  sendChat: (request: ChatRequest) => ipcRenderer.invoke("chat:send", request),
  streamChat: (request: ChatRequest, onUpdate: (update: ChatStreamUpdate) => void) => {
    const streamId = `${Date.now()}-${++chatStreamSequence}`;
    return new Promise<ChatResponse>((resolve, reject) => {
      const listener = (_event: IpcRendererEvent, receivedId: string, update: ChatStreamUpdate) => {
        if (receivedId === streamId) onUpdate(update);
      };
      ipcRenderer.on("chat:stream:update", listener);
      void ipcRenderer.invoke("chat:stream", streamId, request)
        .then((response: ChatResponse) => resolve(response), reject)
        .finally(() => ipcRenderer.removeListener("chat:stream:update", listener));
    });
  },
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
