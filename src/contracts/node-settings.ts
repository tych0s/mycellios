import type { ComputeMode } from "./types.js";

export type CoordinatorMode = "local" | "remote";
export type ComponentUpdateChannel = "stable" | "dev";
export type ArtifactSeedingNetworkPolicy = "unmetered-only" | "any-network";

export interface NodeControlSettings {
  coordinatorMode: CoordinatorMode;
  remoteCoordinatorUrl: string;
  remoteCoordinatorToken: string;
  contributionEnabled: boolean;
  artifactSeedingEnabled: boolean;
  artifactSeedingNetworkPolicy: ArtifactSeedingNetworkPolicy;
  artifactSeedingUploadLimitBytesPerHour: number;
  computeMode: ComputeMode;
  launchAtLogin: boolean;
  closeToTray: boolean;
  onboardingComplete: boolean;
  region: string;
  offeredVramMb: number;
  componentUpdateChannel: ComponentUpdateChannel;
  componentUpdateFeedUrl: string;
  componentUpdateKeyId: string;
  /** Ed25519 SPKI DER encoded as canonical base64url. This is public trust data. */
  componentUpdatePublicKey: string;
}

export interface DeveloperChannelEnrollment {
  feedUrl: string;
  keyId: string;
  /** Ed25519 SPKI DER encoded as canonical base64url. */
  publicKey: string;
}

export const PUBLIC_COORDINATOR_URL = "https://www.mycellios.com";

export const DEFAULT_NODE_SETTINGS: NodeControlSettings = Object.freeze({
  coordinatorMode: "remote",
  remoteCoordinatorUrl: PUBLIC_COORDINATOR_URL,
  remoteCoordinatorToken: "",
  contributionEnabled: false,
  artifactSeedingEnabled: false,
  artifactSeedingNetworkPolicy: "unmetered-only",
  artifactSeedingUploadLimitBytesPerHour: 2 * 1_024 * 1_024 * 1_024,
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
