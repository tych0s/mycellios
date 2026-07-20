import type { DesktopBridge } from "../../desktop/contracts";

declare global {
  interface Window {
    mycellios: DesktopBridge;
  }
}

export {};
