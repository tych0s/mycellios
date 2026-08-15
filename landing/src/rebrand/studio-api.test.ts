import { afterEach, describe, expect, it, vi } from "vitest";
import { draftFromTemplate } from "./studio-model";
import { importStudioDraft, publishStudioAgent } from "./studio-api";

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
});
