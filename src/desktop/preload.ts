import { contextBridge, ipcRenderer } from "electron";
import type {
  ChatRequest,
  DesktopBridge,
  DesktopSettings,
} from "./contracts.js";

const bridge: DesktopBridge = Object.freeze({
  getSnapshot: () => ipcRenderer.invoke("dashboard:read"),
  saveSettings: (settings: DesktopSettings) => ipcRenderer.invoke("settings:save", settings),
  setContribution: (enabled: boolean) => ipcRenderer.invoke("contribution:set", enabled),
  sendChat: (request: ChatRequest) => ipcRenderer.invoke("chat:send", request),
  checkForUpdates: () => ipcRenderer.invoke("updates:check"),
  installUpdate: () => ipcRenderer.invoke("updates:install"),
  minimizeWindow: () => ipcRenderer.invoke("window:minimize"),
  toggleMaximizeWindow: () => ipcRenderer.invoke("window:toggle-maximize"),
  closeWindow: () => ipcRenderer.invoke("window:close"),
});

contextBridge.exposeInMainWorld("mycellios", bridge);
