import type { AuthSession } from "../auth";
import type { StudioAgentConfiguration } from "../../../src/contracts/studio.js";
import type { StudioDraft } from "./studio-model";

export interface SavedStudioAgent { id: string; draftVersion: number; operationalState: string; publishedRevisionId: string | null; configuration?: StudioAgentConfiguration; }

export class StudioApiError extends Error {
  constructor(message: string, readonly status: number, readonly code: string | null) { super(message); }
}

export function studioAgentMatchesDraft(agent: SavedStudioAgent, draft: StudioDraft): boolean {
  return agent.configuration !== undefined && JSON.stringify(agent.configuration) === JSON.stringify(configurationFromDraft(draft));
}

export async function loadStudioAgent(session: AuthSession, agentId: string): Promise<SavedStudioAgent> {
  const response = await studioFetch(`/v1/studio/agents/${encodeURIComponent(agentId)}`, {
    headers: { authorization: `Bearer ${session.accessToken}` },
  });
  const payload = await studioResponse<{ agent?: SavedStudioAgent }>(response);
  return requireStudioAgent(payload.agent, agentId);
}

export async function importStudioDraft(session: AuthSession, draft: StudioDraft, idempotencyKey: string): Promise<SavedStudioAgent> {
  const response = await studioFetch("/v1/studio/agents", {
    method: "POST",
    headers: { authorization: `Bearer ${session.accessToken}`, "content-type": "application/json" },
    body: JSON.stringify({ idempotencyKey, templateId: draft.templateId, configuration: configurationFromDraft(draft) }),
  });
  return requireStudioAgent(await studioResponse(response));
}

export async function saveStudioDraft(session: AuthSession, draft: StudioDraft, agent: SavedStudioAgent | null, idempotencyKey: string): Promise<SavedStudioAgent> {
  if (!agent) return importStudioDraft(session, draft, idempotencyKey);
  try {
    const response = await studioFetch(`/v1/studio/agents/${encodeURIComponent(agent.id)}/draft`, {
      method: "PATCH",
      headers: { authorization: `Bearer ${session.accessToken}`, "content-type": "application/json" },
      body: JSON.stringify({ expectedVersion: agent.draftVersion, configuration: configurationFromDraft(draft) }),
    });
    return requireStudioAgent(await studioResponse(response), agent.id);
  } catch (error) {
    // A timed-out PATCH may have committed. Check the owned record before a retry.
    try {
      const current = await loadStudioAgent(session, agent.id);
      if (studioAgentMatchesDraft(current, draft)) return current;
    } catch { /* Preserve the original failure when reconciliation is unavailable. */ }
    throw error;
  }
}

export async function publishStudioAgent(session: AuthSession, agent: SavedStudioAgent, channel: StudioDraft["channel"], idempotencyKey: string): Promise<{ agent: SavedStudioAgent; deployments: Array<{ id: string; channel: string; state: string; publicId: string }> }> {
  const response = await studioFetch(`/v1/studio/agents/${encodeURIComponent(agent.id)}/publish`, {
    method: "POST", headers: { authorization: `Bearer ${session.accessToken}`, "content-type": "application/json" },
    body: JSON.stringify({ idempotencyKey, expectedVersion: agent.draftVersion, channels: [channel] }),
  });
  const payload = await studioResponse<{ agent?: SavedStudioAgent; deployments?: Array<{ id: string; channel: string; state: string; publicId: string }> }>(response);
  return { agent: requireStudioAgent(payload.agent, agent.id), deployments: Array.isArray(payload.deployments) ? payload.deployments : [] };
}

function configurationFromDraft(draft: StudioDraft) {
  return { name: draft.name, role: draft.role, instructions: draft.instructions, memoryMode: draft.memoryMode, knowledgeSourceIds: [], tools: draft.tools, modelPolicy: { preferredModel: "Qwen/Qwen3-0.6B", fallbackModel: null, privacy: "trusted-only", maxOutputTokens: 512, deadlineMs: 120_000 } };
}

async function studioFetch(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, { ...init, signal: AbortSignal.timeout(30_000) });
  } catch (error) {
    if (error instanceof Error && error.name === "TimeoutError") {
      throw new Error("Studio did not confirm the request in time. It may have saved; retry to check its status.");
    }
    throw error;
  }
}

function requireStudioAgent(value: unknown, expectedId?: string): SavedStudioAgent {
  if (!value || typeof value !== "object") throw new Error("Studio returned an invalid agent. Your local draft is still available.");
  const agent = value as Partial<SavedStudioAgent>;
  if (typeof agent.id !== "string" || !agent.id.startsWith("agt_") || (expectedId && agent.id !== expectedId)
    || !Number.isInteger(agent.draftVersion) || (agent.draftVersion ?? 0) < 1
    || typeof agent.operationalState !== "string"
    || !(agent.publishedRevisionId === null || typeof agent.publishedRevisionId === "string")) {
    throw new Error("Studio returned an invalid agent. Your local draft is still available.");
  }
  return agent as SavedStudioAgent;
}

async function studioResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null) as { error?: { code?: string; message?: string } } | null;
  if (!response.ok) throw new StudioApiError(payload?.error?.message ?? payload?.error?.code ?? `Studio request failed (${response.status}).`, response.status, payload?.error?.code ?? null);
  if (!payload || typeof payload !== "object") throw new Error("Studio returned an invalid response. Your local draft is still available.");
  return payload as T;
}
