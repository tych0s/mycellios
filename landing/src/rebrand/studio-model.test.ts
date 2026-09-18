import { afterEach, describe, expect, it, vi } from "vitest";
import { draftFromTemplate, loadLocalStudioDraft, persistLocalStudioDraft, previewReply, restoreStudioDraft, studioCompletion } from "./studio-model";

afterEach(() => vi.unstubAllGlobals());

describe("Mycellios Studio draft model", () => {
  it("creates isolated template drafts", () => {
    const first = draftFromTemplate("researcher");
    first.knowledge.push("Private notes");
    expect(draftFromTemplate("researcher").knowledge).not.toContain("Private notes");
    expect(draftFromTemplate("developer").channel).toBe("api");
  });

  it("restores only bounded, supported local values", () => {
    const restored = restoreStudioDraft(JSON.stringify({
      templateId: "researcher",
      name: "  Atlas  ",
      role: "Research partner",
      instructions: "Use evidence and identify uncertainty before every conclusion.",
      memoryMode: "unsupported",
      knowledge: ["Source A", 9, "Source B"],
      tools: ["web", "shell", "web"],
      channel: "telegram",
    }));
    expect(restored.name).toBe("Atlas");
    expect(restored.memoryMode).toBe("continuous");
    expect(restored.knowledge).toEqual(["Source A", "Source B"]);
    expect(restored.tools).toEqual(["web"]);
    expect(restored.channel).toBe("telegram");
  });

  it("falls back safely when storage is malformed", () => {
    expect(restoreStudioDraft("not-json").templateId).toBe("concierge");
  });

  it("keeps Studio usable when browser storage is denied or full", () => {
    vi.stubGlobal("window", { get localStorage() { throw new DOMException("Blocked", "SecurityError"); } });
    expect(loadLocalStudioDraft()).toEqual(draftFromTemplate("concierge"));
    expect(persistLocalStudioDraft(draftFromTemplate("concierge"))).toBe(false);
    vi.stubGlobal("window", { localStorage: { getItem: () => null, setItem: () => { throw new DOMException("Full", "QuotaExceededError"); } } });
    expect(persistLocalStudioDraft(draftFromTemplate("researcher"))).toBe(false);
  });

  it("tracks readiness and produces draft-aware preview copy", () => {
    const draft = draftFromTemplate("researcher");
    expect(studioCompletion(draft)).toBe(100);
    expect(previewReply(draft, "What changed?")).toContain("Research library and Decision log");
    expect(previewReply(draft, "What changed?")).toContain("time-sensitive");
    expect(previewReply(draft, "")).toContain("Ask Aster");
  });
});
