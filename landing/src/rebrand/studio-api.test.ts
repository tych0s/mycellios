import { afterEach, describe, expect, it, vi } from "vitest";
import { draftFromTemplate } from "./studio-model";
import { importStudioDraft, publishStudioAgent, saveStudioDraft } from "./studio-api";

afterEach(() => vi.unstubAllGlobals());
const session = { accessToken: "account-token", refreshToken: "refresh", expiresAt: Date.now() + 60_000, user: { id: "owner", email: null } };

describe("Studio authenticated API client", () => {
  it("imports a non-secret normalized draft explicitly", async () => {
    const request = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => Response.json({ id: "agt_1", draftVersion: 1, operationalState: "draft", publishedRevisionId: null }, { status: 201 }));
    vi.stubGlobal("fetch", request);
    await importStudioDraft(session, draftFromTemplate("concierge"), "import-1");
    expect(request).toHaveBeenCalledWith("/v1/studio/agents", expect.objectContaining({ method: "POST", headers: expect.objectContaining({ authorization: "Bearer account-token" }) }));
    const body = JSON.parse(request.mock.calls[0]![1]!.body as string);
    expect(body.configuration).toMatchObject({ knowledgeSourceIds: [], modelPolicy: { privacy: "trusted-only" } });
    expect(JSON.stringify(body)).not.toMatch(/refresh|account-token/);
  });

  it("surfaces the real waiting state returned by publication", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ agent: { id: "agt_1", draftVersion: 1, operationalState: "waiting_for_capacity", publishedRevisionId: "rev_1" }, deployments: [{ id: "dep_1", channel: "web", state: "waiting_for_capacity", publicId: "agent_public" }] })));
    await expect(publishStudioAgent(session, { id: "agt_1", draftVersion: 1, operationalState: "draft", publishedRevisionId: null }, "web", "publish-1")).resolves.toMatchObject({ agent: { operationalState: "waiting_for_capacity" } });
  });

  it("saves edited configuration and publishes the returned draft version", async () => {
    const request = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => init?.method === "PATCH"
      ? Response.json({ id: "agt_1", draftVersion: 2, operationalState: "draft", publishedRevisionId: "rev_1" })
      : Response.json({ agent: { id: "agt_1", draftVersion: 2, operationalState: "waiting_for_capacity", publishedRevisionId: "rev_2" }, deployments: [] }));
    vi.stubGlobal("fetch", request);
    const draft = { ...draftFromTemplate("researcher"), name: "Revised name", instructions: "Use only the revised approved sources." };
    const saved = await saveStudioDraft(session, draft, { id: "agt_1", draftVersion: 1, operationalState: "waiting_for_capacity", publishedRevisionId: "rev_1" }, "unused-create-key");
    await publishStudioAgent(session, saved, draft.channel, "publish-2");
    expect(request.mock.calls[0]?.[0]).toBe("/v1/studio/agents/agt_1/draft");
    expect(JSON.parse(request.mock.calls[0]![1]!.body as string)).toMatchObject({ expectedVersion: 1, configuration: { name: "Revised name", instructions: "Use only the revised approved sources." } });
    expect(JSON.parse(request.mock.calls[1]![1]!.body as string)).toMatchObject({ expectedVersion: 2, channels: ["web"] });
  });

  it("rejects a conflicting edit before a stale revision can be published", async () => {
    const request = vi.fn(async () => Response.json({ error: { code: "studio_agent_version_conflict", message: "The Studio draft changed in another session." } }, { status: 409 }));
    vi.stubGlobal("fetch", request);
    await expect(saveStudioDraft(session, draftFromTemplate("concierge"), { id: "agt_1", draftVersion: 1, operationalState: "draft", publishedRevisionId: null }, "unused-key")).rejects.toThrow("another session");
    expect(request).toHaveBeenCalledOnce();
  });

  it("reports malformed success responses as errors", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("not-json", { status: 200 })));
    await expect(importStudioDraft(session, draftFromTemplate("concierge"), "import-key")).rejects.toThrow("invalid response");
  });
});
