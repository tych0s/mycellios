import type { AuthSession } from "../auth";
import type { StudioDraft } from "./studio-model";

export interface SavedStudioAgent { id: string; draftVersion: number; operationalState: string; publishedRevisionId: string | null; }

export async function importStudioDraft(session: AuthSession, draft: StudioDraft, idempotencyKey: string): Promise<SavedStudioAgent> {
  const response = await fetch("/v1/studio/agents", {
    method: "POST",
    headers: { authorization: `Bearer ${session.accessToken}`, "content-type": "application/json" },
    body: JSON.stringify({ idempotencyKey, templateId: draft.templateId, configuration: configurationFromDraft(draft) }),
  });
  return studioResponse(response);
}

export async function saveStudioDraft(session: AuthSession, draft: StudioDraft, agent: SavedStudioAgent | null, idempotencyKey: string): Promise<SavedStudioAgent> {
  if (!agent) return importStudioDraft(session, draft, idempotencyKey);
  const response = await fetch(`/v1/studio/agents/${encodeURIComponent(agent.id)}/draft`, {
    method: "PATCH",
    headers: { authorization: `Bearer ${session.accessToken}`, "content-type": "application/json" },
    body: JSON.stringify({ expectedVersion: agent.draftVersion, configuration: configurationFromDraft(draft) }),
  });
  return studioResponse(response);
}

export async function publishStudioAgent(session: AuthSession, agent: SavedStudioAgent, channel: StudioDraft["channel"], idempotencyKey: string): Promise<{ agent: SavedStudioAgent; deployments: Array<{ id: string; channel: string; state: string; publicId: string }> }> {
  const response = await fetch(`/v1/studio/agents/${encodeURIComponent(agent.id)}/publish`, {
    method: "POST", headers: { authorization: `Bearer ${session.accessToken}`, "content-type": "application/json" },
    body: JSON.stringify({ idempotencyKey, expectedVersion: agent.draftVersion, channels: [channel] }),
  });
  return studioResponse(response);
}

function configurationFromDraft(draft: StudioDraft) {
  return { name: draft.name, role: draft.role, instructions: draft.instructions, memoryMode: draft.memoryMode, knowledgeSourceIds: [], tools: draft.tools, modelPolicy: { preferredModel: "Qwen/Qwen3-0.6B", fallbackModel: null, privacy: "trusted-only", maxOutputTokens: 512, deadlineMs: 120_000 } };
}

async function studioResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null) as { error?: { code?: string; message?: string } } | null;
  if (!response.ok) throw new Error(payload?.error?.message ?? payload?.error?.code ?? `Studio request failed (${response.status}).`);
  if (!payload || typeof payload !== "object") throw new Error("Studio returned an invalid response. Your local draft is still available.");
  return payload as T;
}
