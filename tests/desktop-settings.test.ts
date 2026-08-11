import { describe, expect, it } from "vitest";
import { generateKeyPairSync } from "node:crypto";
import {
  DEFAULT_DESKTOP_SETTINGS,
  componentUpdateTrustMatches,
  desktopSettingsRequireMigration,
  sanitizeDeveloperChannelEnrollment,
  sanitizeDesktopSettings,
} from "../src/node/settings.js";

describe("native-only desktop settings", () => {
  const devPublicKey = generateKeyPairSync("ed25519").publicKey.export({
    format: "der",
    type: "spki",
  }).toString("base64url");

  it("migrates legacy local model runtime settings without retaining an external runtime route", () => {
    const stored = {
      ...DEFAULT_DESKTOP_SETTINGS,
      remoteCoordinatorUrl: "https://mycellios.com/",
      contributionEnabled: true,
      adapterMode: "local-model-runtime",
      modelName: "qwen3:0.6b",
      adapterBaseUrl: "http://127.0.0.1:11434",
      modelDigest: "sha256:legacy",
    };

    const migrated = sanitizeDesktopSettings(stored);

    expect(migrated).toEqual({
      ...DEFAULT_DESKTOP_SETTINGS,
      contributionEnabled: true,
    });
    expect(migrated).not.toHaveProperty("adapterMode");
    expect(migrated).not.toHaveProperty("modelName");
    expect(migrated).not.toHaveProperty("adapterBaseUrl");
    expect(migrated).not.toHaveProperty("modelDigest");
    expect(desktopSettingsRequireMigration(stored, migrated)).toBe(true);
  });

  it("keeps current settings stable and clamps the offered memory budget", () => {
    const current = {
      ...DEFAULT_DESKTOP_SETTINGS,
      coordinatorMode: "local" as const,
      computeMode: "gpu-only" as const,
      offeredVramMb: 999_999,
    };

    const sanitized = sanitizeDesktopSettings(current);

    expect(sanitized.offeredVramMb).toBe(262_144);
    expect(sanitized.computeMode).toBe("gpu-only");
    expect(desktopSettingsRequireMigration(sanitized, sanitized)).toBe(false);
  });

  it("continues to reject an insecure non-loopback coordinator", () => {
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      remoteCoordinatorUrl: "http://network.example.test",
    })).toThrow("Remote coordinators must use HTTPS/WSS");
  });

  it("migrates settings created before component channels to stable defaults", () => {
    const {
      componentUpdateChannel: _channel,
      componentUpdateFeedUrl: _feed,
      componentUpdateKeyId: _keyId,
      componentUpdatePublicKey: _publicKey,
      ...legacy
    } = DEFAULT_DESKTOP_SETTINGS;

    expect(sanitizeDesktopSettings(legacy)).toEqual(DEFAULT_DESKTOP_SETTINGS);
  });

  it("accepts an explicitly enrolled Ed25519 dev feed on loopback", () => {
    const sanitized = sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateFeedUrl: "http://127.0.0.1:8787/",
      componentUpdateKeyId: "mycellios-dev-local-1",
      componentUpdatePublicKey: devPublicKey,
    });

    expect(sanitized.componentUpdateChannel).toBe("dev");
    expect(sanitized.componentUpdateFeedUrl).toBe("http://127.0.0.1:8787");
    expect(sanitized.componentUpdatePublicKey).toBe(devPublicKey);
  });

  it("validates a dedicated dev enrollment independently from generic settings", () => {
    expect(sanitizeDeveloperChannelEnrollment({
      feedUrl: "https://updates.example.test/",
      keyId: "mycellios-dev-1",
      publicKey: devPublicKey,
    })).toEqual({
      feedUrl: "https://updates.example.test",
      keyId: "mycellios-dev-1",
      publicKey: devPublicKey,
    });
    expect(() => sanitizeDeveloperChannelEnrollment({
      feedUrl: "http://updates.example.test",
      keyId: "mycellios-dev-1",
      publicKey: devPublicKey,
    })).toThrow("Component updates require HTTPS");
    expect(() => sanitizeDeveloperChannelEnrollment({
      feedUrl: `https://updates.example.test/${"a".repeat(4_096)}`,
      keyId: "mycellios-dev-1",
      publicKey: devPublicKey,
    })).toThrow("feed URL is invalid");
  });

  it("detects every component trust change before generic settings are saved", () => {
    const enrolled = sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateFeedUrl: "https://updates.example.test",
      componentUpdateKeyId: "mycellios-dev-1",
      componentUpdatePublicKey: devPublicKey,
    });
    expect(componentUpdateTrustMatches(enrolled, { ...enrolled })).toBe(true);
    expect(componentUpdateTrustMatches(enrolled, {
      ...enrolled,
      componentUpdatePublicKey: generateKeyPairSync("ed25519").publicKey.export({
        format: "der",
        type: "spki",
      }).toString("base64url"),
    })).toBe(false);
    expect(componentUpdateTrustMatches(enrolled, {
      ...enrolled,
      componentUpdateFeedUrl: "https://attacker.example.test",
    })).toBe(false);
    expect(componentUpdateTrustMatches(enrolled, {
      ...enrolled,
      componentUpdateChannel: "stable",
    })).toBe(false);
  });

  it("rejects unknown channels and feeds the updater cannot consume", () => {
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "preview",
    })).toThrow("component update channel is invalid");
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateFeedUrl: "wss://updates.example.test",
      componentUpdateKeyId: "mycellios-dev-local-1",
      componentUpdatePublicKey: devPublicKey,
    })).toThrow("Component updates require HTTPS");
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateFeedUrl: "http://updates.example.test",
      componentUpdateKeyId: "mycellios-dev-local-1",
      componentUpdatePublicKey: devPublicKey,
    })).toThrow("Component updates require HTTPS");
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateFeedUrl: "https://updates.example.test/?token=secret",
      componentUpdateKeyId: "mycellios-dev-local-1",
      componentUpdatePublicKey: devPublicKey,
    })).toThrow("Component updates require HTTPS");
  });

  it("rejects incomplete or non-Ed25519 dev trust enrollment", () => {
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateKeyId: "mycellios-dev-local-1",
    })).toThrow("component update trust key is invalid");
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateKeyId: "mycellios-dev-local-1",
      componentUpdatePublicKey: Buffer.from("not-an-spki").toString("base64url"),
    })).toThrow("component update trust key is invalid");
    expect(() => sanitizeDesktopSettings({
      ...DEFAULT_DESKTOP_SETTINGS,
      componentUpdateChannel: "dev",
      componentUpdateKeyId: "mycellios-dev-local-1",
      componentUpdatePublicKey: Buffer.concat([
        Buffer.from(devPublicKey, "base64url"),
        Buffer.from([0]),
      ]).toString("base64url"),
    })).toThrow("component update trust key is invalid");
  });
});
