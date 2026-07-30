import { describe, expect, it } from "vitest";
import type { DashboardSnapshot } from "../../src/desktop/contracts";
import { desktopToPublicSnapshot } from "./Panel";

function desktopSnapshot(coordinatorMode: "local" | "remote"): DashboardSnapshot {
  return {
    capturedAt: "2026-07-30T09:00:00.000Z",
    health: null,
    workers: [],
    models: [],
    requestedModels: [],
    jobs: [],
    settings: { coordinatorMode },
  } as unknown as DashboardSnapshot;
}

describe("desktop public snapshot federation", () => {
  it("preserves the remote federation inventory while applying desktop state", () => {
    const source = {
      federation: {
        enabled: true,
        readyNetworks: 1,
        routableNodes: 1,
        verifiedModels: 1,
      },
      federatedNetworks: [{ id: "ai-horde", actualState: "ready" }],
      federatedNodes: [{ id: "fed_worker", networkId: "ai-horde", routable: true }],
    } as unknown as NonNullable<Parameters<typeof desktopToPublicSnapshot>[1]>;

    const result = desktopToPublicSnapshot(desktopSnapshot("remote"), source);

    expect(result.federation.enabled).toBe(true);
    expect(result.federatedNetworks).toEqual(source.federatedNetworks);
    expect(result.federatedNodes).toEqual(source.federatedNodes);
  });

  it("does not invent federation for a local desktop snapshot", () => {
    const result = desktopToPublicSnapshot(desktopSnapshot("local"));

    expect(result.federation.enabled).toBe(false);
    expect(result.federatedNetworks).toEqual([]);
    expect(result.federatedNodes).toEqual([]);
  });
});
