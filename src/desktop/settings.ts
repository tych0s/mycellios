import type { DesktopSettings } from "./contracts.js";
import { normalizeComputeMode } from "./compute-mode.js";

export const PUBLIC_COORDINATOR_URL = "https://www.mycellios.com";

export const DEFAULT_DESKTOP_SETTINGS: DesktopSettings = Object.freeze({
  coordinatorMode: "remote",
  remoteCoordinatorUrl: PUBLIC_COORDINATOR_URL,
  remoteCoordinatorToken: "",
  contributionEnabled: false,
  computeMode: "automatic",
  launchAtLogin: false,
  closeToTray: true,
  onboardingComplete: false,
  region: "auto",
  offeredVramMb: 4_096,
});

const LEGACY_PUBLIC_COORDINATOR_URLS = new Set([
  "https://www.mycellios.com",
  "https://mycellios.com",
  "https://network.mycellios.app",
]);
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

/**
 * Normalizes both current settings and pre-native desktop settings.
 *
 * Legacy adapter fields are deliberately not copied to the result. Old
 * local model runtime/connectivity-test selections therefore migrate to a hardware-only
 * Mycellios node, without ever activating the old inference path.
 */
export function sanitizeDesktopSettings(input: unknown): DesktopSettings {
  const source = record(input);
  const coordinatorMode = source.coordinatorMode === "local" ? "local" : "remote";
  const storedCoordinatorUrl = text(source.remoteCoordinatorUrl).replace(/\/+$/, "");
  const remoteCoordinatorUrl = LEGACY_PUBLIC_COORDINATOR_URLS.has(storedCoordinatorUrl)
    ? DEFAULT_DESKTOP_SETTINGS.remoteCoordinatorUrl
    : storedCoordinatorUrl || DEFAULT_DESKTOP_SETTINGS.remoteCoordinatorUrl;
  if (coordinatorMode === "remote") validateDesktopCoordinatorUrl(remoteCoordinatorUrl);

  const offeredVram = typeof source.offeredVramMb === "number"
    ? source.offeredVramMb
    : Number.NaN;
  const offeredVramMb = Number.isFinite(offeredVram)
    ? Math.max(512, Math.min(262_144, Math.round(offeredVram)))
    : DEFAULT_DESKTOP_SETTINGS.offeredVramMb;

  return {
    coordinatorMode,
    remoteCoordinatorUrl,
    remoteCoordinatorToken: text(source.remoteCoordinatorToken),
    contributionEnabled: source.contributionEnabled === true,
    computeMode: normalizeComputeMode(source.computeMode),
    launchAtLogin: source.launchAtLogin === true,
    closeToTray: source.closeToTray !== false,
    onboardingComplete: source.onboardingComplete === true,
    region: text(source.region) || "auto",
    offeredVramMb,
  };
}

export function desktopSettingsRequireMigration(
  stored: unknown,
  sanitized: DesktopSettings,
): boolean {
  if (!stored || typeof stored !== "object" || Array.isArray(stored)) return true;
  return JSON.stringify(stored) !== JSON.stringify(sanitized);
}

function validateDesktopCoordinatorUrl(raw: string): void {
  const normalized = raw.endsWith("/") ? raw : `${raw}/`;
  const url = new URL(normalized);
  if (!new Set(["http:", "https:", "ws:", "wss:"]).has(url.protocol)) {
    throw new Error("Coordinator URL must use HTTP(S) or WS(S)");
  }
  if (
    new Set(["http:", "ws:"]).has(url.protocol)
    && !LOOPBACK_HOSTS.has(url.hostname.toLowerCase())
  ) {
    throw new Error("Remote coordinators must use HTTPS/WSS");
  }
  if (url.username || url.password) {
    throw new Error("Coordinator URL must not embed credentials");
  }
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}
