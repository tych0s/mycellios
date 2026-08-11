import { describe, expect, it } from "vitest";
import { localOperationalState, nodeOperationalState, operationalState, type LocalOperationalInput } from "./operational-state";

const observed = { state: "ready", contributionEnabled: true, draining: false } as const;

describe("product operational states", () => {
  it.each([
    ["revoked", { status: "revoked", connected: false, observed, latestCommand: null }],
    ["rollback", { status: "active", connected: true, observed, latestCommand: { type: "rollback", state: "delivered" } }],
    ["updating", { status: "active", connected: true, observed, latestCommand: { type: "update", state: "queued" } }],
    ["pairing", { status: "active", connected: false, observed: null, latestCommand: null }],
    ["canarying", { status: "active", connected: true, observed: { ...observed, state: "canary" }, latestCommand: null }],
    ["failed", { status: "active", connected: true, observed: { ...observed, state: "failed" }, latestCommand: null }],
    ["degraded", { status: "active", connected: true, observed: { ...observed, state: "degraded" }, latestCommand: null }],
    ["reconnecting", { status: "active", connected: false, observed, latestCommand: null }],
    ["draining", { status: "active", connected: true, observed: { ...observed, draining: true }, latestCommand: null }],
    ["paused", { status: "active", connected: true, observed: { ...observed, contributionEnabled: false }, latestCommand: null }],
    ["ready", { status: "active", connected: true, observed, latestCommand: null }],
  ] as const)("prioritizes %s without presenting a false ready state", (expected, input) => {
    expect(nodeOperationalState(input).id).toBe(expected);
  });

  it("covers download and canary preparation from real local lifecycle fields", () => {
    const base: LocalOperationalInput = {
      update: { state: "idle", currentVersion: "1", availableVersion: null, message: "", checkedAt: null },
      acceleration: {
        state: "cpu-ready", requestedBackend: null, effectiveBackend: "cpu", deviceName: "CPU", precision: "float32", message: "",
        cpu: { state: "ready", activeStages: 0, deviceName: "CPU", precision: "float32", message: "" },
        gpu: { state: "not-detected", vendor: null, model: null, backend: null, activeStages: 0 },
        preparation: { phase: "idle", progressPct: null, bytesCompleted: null, bytesTotal: null, bytesPerSecond: null, etaSeconds: null, startedAt: null, updatedAt: null, currentArtifact: null, artifactIndex: null, artifactCount: null, issue: null, log: [] },
      },
      contribution: "connected",
    };
    expect(localOperationalState({ ...base, acceleration: { ...base.acceleration, preparation: { ...base.acceleration.preparation, phase: "downloading" } } }).id).toBe("downloading");
    expect(localOperationalState({ ...base, acceleration: { ...base.acceleration, preparation: { ...base.acceleration.preparation, phase: "physical-probe" } } }).id).toBe("canarying");
  });

  it("gives every required state truthful copy and a next action", () => {
    for (const id of ["pairing", "downloading", "canarying", "ready", "paused", "draining", "reconnecting", "degraded", "failed", "revoked", "updating", "rollback"] as const) {
      expect(operationalState(id)).toMatchObject({ id });
      expect(operationalState(id).summary.length).toBeGreaterThan(20);
      expect(operationalState(id).nextAction.length).toBeGreaterThan(3);
    }
  });
});
