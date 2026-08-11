export type SystemLogLevel = "info" | "warning" | "error";
export interface SystemLogEntry {
  id: string;
  at: string;
  level: SystemLogLevel;
  source: "desktop" | "coordinator" | "worker" | "runtime" | "renderer" | "network";
  event: string;
  message: string;
  details?: string;
}
export interface SystemLogSnapshot {
  capturedAt: string;
  entries: SystemLogEntry[];
  truncated: boolean;
  source: "desktop-file" | "network-snapshot";
}
