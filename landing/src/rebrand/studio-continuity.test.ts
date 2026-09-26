import { afterEach, describe, expect, it, vi } from "vitest";
import { draftFromTemplate } from "./studio-model";
import { loadStudioContinuity, saveStudioContinuity } from "./studio-continuity";

afterEach(() => vi.unstubAllGlobals());

describe("Studio publication continuity", () => {
  it("persists the original import and publication keys per account", async () => {
    const records = new Map<string, string>();
    vi.stubGlobal("window", { localStorage: {
      getItem: (key: string) => records.get(key) ?? null,
      setItem: (key: string, value: string) => { records.set(key, value); },
    } });
    const owner = "owner-a-continuity";
    const state = {
      agentId: "agt_123",
      create: { key: "import-123", draft: draftFromTemplate("researcher") },
      publish: { key: "publish-123", agentId: "agt_123", draftVersion: 4, channel: "web" as const },
    };
    expect(saveStudioContinuity(owner, state)).toBe(true);
    expect(loadStudioContinuity(owner)).toEqual(state);
    expect(loadStudioContinuity("owner-b-continuity")).toEqual({});
    expect([...records.keys()]).toEqual(["mycellios.studio.remote.v1.owner-a-continuity"]);
    vi.resetModules();
    const afterReload = await import("./studio-continuity");
    expect(afterReload.loadStudioContinuity(owner)).toEqual(state);
  });

  it("ignores malformed persisted pending operations", () => {
    vi.stubGlobal("window", { localStorage: {
      getItem: () => JSON.stringify({ agentId: "invalid", create: { key: "bad", draft: {} }, publish: { key: "bad", channel: "web" } }),
    } });
    expect(loadStudioContinuity("owner-malformed-continuity")).toEqual({});
  });
});
