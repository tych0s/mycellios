import { describe, expect, it } from "vitest";
import {
  engineRuntimeProfileNeedsChallenge,
  activeEngineRuntimeActivationPlans,
  ENGINE_RUNTIME_PROFILE_RENEWAL_LEAD_MS,
  engineRuntimeActivationPlansReady,
} from "../src/coordinator/engine-runtime-profile-scheduler.js";
import { sealEngineRuntimeProfile } from "../src/contracts/engine-runtime-profile.js";
import type { EngineRuntimeChallengeRequest } from "../src/coordinator/worker-hub.js";
import { WorkerHub } from "../src/coordinator/worker-hub.js";

const digest = (value: string) => `sha256:${value.repeat(64)}` as const;
const now = Date.parse("2026-08-10T12:00:00.000Z");

function request(): EngineRuntimeChallengeRequest {
  return {
    probeKind: "qwen3-dense-v1",
    descriptorDigest: digest("a"), certificationId: digest("b"),
    artifactManifestDigest: digest("c"), modelId: "qwen-test",
    modelRevision: "1".repeat(40), backend: "cuda", runtimeAbi: "cuda-12",
    quantization: "bf16", contextTokens: 8_192,
    expectedLayerStart: 0, expectedLayerEnd: 16,
    expectedKvBytesPerToken: 32_768, expectedLayerWeightBytes: 1_024,
    referenceDecodeMsPerToken: 8, referencePrefillMsPerToken: 0.4,
    hiddenSize: 4_096, attentionHeads: 32, kvHeads: 4, headDim: 128,
    requiredRoles: ["head", "tail"], minimumSamples: 7,
  };
}

function profile(expiresAt: number, overrides: {
  measuredAt?: number;
  confidenceHalfWidthPct?: number;
  workerId?: string;
} = {}) {
  return sealEngineRuntimeProfile({
    descriptorDigest: digest("a"), certificationId: digest("b"),
    artifactManifestDigest: digest("c"), sourceId: digest("d"),
    hardwareFingerprintSha256: digest("e"), workerId: overrides.workerId ?? "worker-1",
    sessionId: "session-1", nodeId: "node-1", modelId: "qwen-test",
    modelRevision: "1".repeat(40), backend: "cuda", runtimeAbi: "cuda-12",
    quantization: "bf16", measuredAt: new Date(overrides.measuredAt ?? now - 60_000).toISOString(),
    expiresAt: new Date(expiresAt).toISOString(), samples: 21,
    confidenceHalfWidthPct: overrides.confidenceHalfWidthPct ?? 5,
    capacity: { contextTokens: 8_192, maxLayerCount: 16, kvBytesPerToken: 32_768,
      maxKvTokens: 16_384, usableMemoryBytes: 16_000_000_000 },
    costs: { decodeMsPerTokenP50: 4, decodeMsPerTokenP95: 5,
      prefillMsPerTokenP50: 0.2, prefillMsPerTokenP95: 0.3,
      verifyMsPerTokenP50: 3, verifyMsPerTokenP95: 4,
      decodeScale: 0.5, prefillScale: 0.5 },
    features: { fastKernel: true, graphMode: "available", roles: ["head", "middle", "tail"] },
    evidence: { deploymentCanaryEvidenceId: digest("f"), runtimePerformanceEvidenceId: digest("0") },
  });
}

describe("engine runtime profile renewal scheduler", () => {
  it("requests the first profile and renews before expiry", () => {
    expect(engineRuntimeProfileNeedsChallenge([], request(), now)).toBe(true);
    expect(engineRuntimeProfileNeedsChallenge([
      profile(now + ENGINE_RUNTIME_PROFILE_RENEWAL_LEAD_MS),
    ], request(), now)).toBe(true);
  });

  it("keeps a fresh authority-matching profile", () => {
    expect(engineRuntimeProfileNeedsChallenge([
      profile(now + ENGINE_RUNTIME_PROFILE_RENEWAL_LEAD_MS + 1),
    ], request(), now)).toBe(false);
  });

  it("renews noisy, stale, and future-dated evidence before nominal expiry", () => {
    const farFutureExpiry = now + 48 * 60 * 60_000;
    expect(engineRuntimeProfileNeedsChallenge([
      profile(farFutureExpiry, { confidenceHalfWidthPct: 21 }),
    ], request(), now)).toBe(true);
    expect(engineRuntimeProfileNeedsChallenge([
      profile(farFutureExpiry, { measuredAt: now - 24 * 60 * 60_000 - 1 }),
    ], request(), now)).toBe(true);
    expect(engineRuntimeProfileNeedsChallenge([
      profile(farFutureExpiry, { measuredAt: now + 60_001 }),
    ], request(), now)).toBe(true);
  });

  it("stops renewing once a fresh low-noise replacement is published", () => {
    const farFutureExpiry = now + 48 * 60 * 60_000;
    const noisy = profile(farFutureExpiry, { confidenceHalfWidthPct: 21 });
    const renewed = profile(farFutureExpiry, { measuredAt: now, confidenceHalfWidthPct: 4 });
    expect(engineRuntimeProfileNeedsChallenge([noisy], request(), now)).toBe(true);
    expect(engineRuntimeProfileNeedsChallenge([noisy, renewed], request(), now)).toBe(false);
  });

  it("renews when authority, exact KV scope, context, layers, or roles differ", () => {
    const fresh = profile(now + 2 * ENGINE_RUNTIME_PROFILE_RENEWAL_LEAD_MS);
    for (const changed of [
      { ...request(), certificationId: digest("9") },
      { ...request(), expectedKvBytesPerToken: 16_384 },
      { ...request(), contextTokens: 32_768 },
      { ...request(), expectedLayerEnd: 17 },
      { ...request(), requiredRoles: ["draft"] as EngineRuntimeChallengeRequest["requiredRoles"] },
    ]) expect(engineRuntimeProfileNeedsChallenge([fresh], changed, now)).toBe(true);
  });

  it("rejects invalid renewal policy", () => {
    expect(() => engineRuntimeProfileNeedsChallenge([], request(), now, -1))
      .toThrow("engine_runtime_profile_renewal_policy_is_invalid");
  });

  it("keeps a bootstrap route private until every exact stage profile exists", () => {
    const worker = (id: string, profiles: ReturnType<typeof profile>[]) => ({
      id,
      capabilities: { distributedExecutor: { engineProfiles: profiles } },
    }) as never;
    const secondRequest = { ...request(), expectedLayerStart: 16, expectedLayerEnd: 32 };
    const plans = [
      { workerId: "worker-1", request: request(), evidence: {
        canaryEvidenceId: digest("f"),
      } },
      { workerId: "worker-2", request: secondRequest, evidence: {
        canaryEvidenceId: digest("f"),
      } },
    ] as never;
    expect(engineRuntimeActivationPlansReady(plans, [
      worker("worker-1", [profile(now + 60_000)]),
      worker("worker-2", []),
    ], now)).toBe(false);
    const secondProfile = profile(now + 60_000, { workerId: "worker-2" });
    expect(engineRuntimeActivationPlansReady(plans, [
      worker("worker-1", [profile(now + 60_000)]),
      worker("worker-2", [secondProfile]),
    ], now)).toBe(true);
  });

  it("requires at least one certified stage before declaring readiness", () => {
    expect(engineRuntimeActivationPlansReady([], [], now)).toBe(false);
  });

  it("renews bootstrap capacity while waiting and resolves after publication", async () => {
    const plan = { workerId: "worker-1", request: request(), evidence: {
      canaryEvidenceId: digest("f"),
    } } as never;
    let workers: Array<{
      id: string;
      capabilities: { distributedExecutor: { engineProfiles: ReturnType<typeof profile>[] } };
    }> = [({
      id: "worker-1",
      capabilities: { distributedExecutor: { engineProfiles: [] } },
    })];
    const store = {
      listRuntimeLinkSamples: () => [],
      listWorkers: () => workers,
      getWorker: (workerId: string) => workers.find((worker) => worker.id === workerId),
    } as never;
    const hub = new WorkerHub(store);
    let renewals = 0;
    await hub.waitForEngineRuntimeProfiles([plan], 100, 1, () => {
      renewals += 1;
      const publishedAt = Date.now();
      workers = [({
        id: "worker-1",
        capabilities: { distributedExecutor: {
          engineProfiles: [profile(publishedAt + 60_000, { measuredAt: publishedAt })],
        } },
      })];
    });
    expect(renewals).toBe(1);
  });

  it("renews only authority tied to the currently committed active route", () => {
    const plan = {
      modelId: "qwen-test",
      routeReservationId: "reservation-1",
    } as never;
    const controller = (observedState: "active" | "degraded", status: "committed" | "released") => ({
      getState: () => ({ desiredState: "active", observedState }),
      listReservations: () => [{ id: "reservation-1", status }],
    }) as never;
    expect(activeEngineRuntimeActivationPlans([plan], controller("active", "committed")))
      .toEqual([plan]);
    expect(activeEngineRuntimeActivationPlans([plan], controller("degraded", "committed")))
      .toEqual([]);
    expect(activeEngineRuntimeActivationPlans([plan], controller("active", "released")))
      .toEqual([]);
  });
});
