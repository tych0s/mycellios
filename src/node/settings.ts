import type {
  DesktopDeveloperChannelEnrollment,
  DesktopSettings,
} from "../contracts/control-api.js";
import { normalizeComputeMode } from "./compute-mode.js";
import { createPublicKey } from "node:crypto";

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
  componentUpdateChannel: "stable",
  componentUpdateFeedUrl: PUBLIC_COORDINATOR_URL,
  componentUpdateKeyId: "",
  componentUpdatePublicKey: "",
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
  if (
    source.componentUpdateChannel !== undefined
    && source.componentUpdateChannel !== "dev"
    && source.componentUpdateChannel !== "stable"
  ) {
    throw new Error("The component update channel is invalid.");
  }
  const componentUpdateChannel = source.componentUpdateChannel === "dev"
    ? "dev"
    : "stable";
  const storedComponentUpdateFeedUrl = text(
    source.componentUpdateFeedUrl,
  ).replace(/\/+$/, "");
  const componentUpdateFeedUrl =
    storedComponentUpdateFeedUrl
    || DEFAULT_DESKTOP_SETTINGS.componentUpdateFeedUrl;
  validateComponentUpdateFeedUrl(componentUpdateFeedUrl);
  const componentUpdateKeyId = text(source.componentUpdateKeyId);
  const componentUpdatePublicKey = text(source.componentUpdatePublicKey);
  const componentTrustValid =
    (componentUpdateKeyId === "" && componentUpdatePublicKey === "")
    || (
      /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(componentUpdateKeyId)
      && validEd25519Spki(componentUpdatePublicKey)
    );
  if (!componentTrustValid) {
    throw new Error("The component update trust key is invalid.");
  }

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
    componentUpdateChannel,
    componentUpdateFeedUrl,
    componentUpdateKeyId,
    componentUpdatePublicKey,
  };
}

export function desktopSettingsRequireMigration(
  stored: unknown,
  sanitized: DesktopSettings,
): boolean {
  if (!stored || typeof stored !== "object" || Array.isArray(stored)) return true;
  return JSON.stringify(stored) !== JSON.stringify(sanitized);
}

export function sanitizeDeveloperChannelEnrollment(
  input: unknown,
): DesktopDeveloperChannelEnrollment {
  const source = record(input);
  const feedUrl = text(source.feedUrl).replace(/\/+$/, "");
  const keyId = text(source.keyId);
  const publicKey = text(source.publicKey);
  validateComponentUpdateFeedUrl(feedUrl);
  if (
    !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(keyId)
    || !validEd25519Spki(publicKey)
  ) {
    throw new Error("The component update trust key is invalid.");
  }
  return { feedUrl, keyId, publicKey };
}

export function componentUpdateTrustMatches(
  current: Pick<
    DesktopSettings,
    | "componentUpdateChannel"
    | "componentUpdateFeedUrl"
    | "componentUpdateKeyId"
    | "componentUpdatePublicKey"
  >,
  next: Pick<
    DesktopSettings,
    | "componentUpdateChannel"
    | "componentUpdateFeedUrl"
    | "componentUpdateKeyId"
    | "componentUpdatePublicKey"
  >,
): boolean {
  return current.componentUpdateChannel === next.componentUpdateChannel
    && current.componentUpdateFeedUrl === next.componentUpdateFeedUrl
    && current.componentUpdateKeyId === next.componentUpdateKeyId
    && current.componentUpdatePublicKey === next.componentUpdatePublicKey;
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

function canonicalBase64url(value: string): boolean {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) return false;
  const decoded = Buffer.from(value, "base64url");
  return decoded.length > 0 && decoded.toString("base64url") === value;
}

function validEd25519Spki(value: string): boolean {
  if (value.length === 0 || value.length > 256) return false;
  if (!canonicalBase64url(value)) return false;
  try {
    const encoded = Buffer.from(value, "base64url");
    if (encoded.length > 128) return false;
    const key = createPublicKey({
      key: encoded,
      format: "der",
      type: "spki",
    });
    const canonical = key.export({ format: "der", type: "spki" });
    return key.asymmetricKeyType === "ed25519"
      && Buffer.isBuffer(canonical)
      && canonical.equals(encoded);
  } catch {
    return false;
  }
}

function validateComponentUpdateFeedUrl(raw: string): void {
  if (raw.length === 0 || raw.length > 4_096) {
    throw new Error("Component update feed URL is invalid.");
  }
  const url = new URL(raw.endsWith("/") ? raw : `${raw}/`);
  const host = url.hostname.toLowerCase();
  const loopback = host === "localhost" || host === "127.0.0.1"
    || host === "::1" || host === "[::1]";
  if (
    (url.protocol !== "https:" && !(url.protocol === "http:" && loopback))
    || url.username
    || url.password
    || url.hash
    || url.search
  ) {
    throw new Error(
      "Component updates require HTTPS, except on explicit loopback development feeds.",
    );
  }
}
