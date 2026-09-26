import type { StudioChannelId, StudioDraft } from "./studio-model";

export interface StudioContinuity {
  agentId?: string;
  create?: { key: string; draft: StudioDraft } | undefined;
  publish?: { key: string; agentId: string; draftVersion: number; channel: StudioChannelId } | undefined;
}

const STORAGE_PREFIX = "mycellios.studio.remote.v1.";
const inMemory = new Map<string, StudioContinuity>();

function storageKey(ownerId: string): string {
  return `${STORAGE_PREFIX}${encodeURIComponent(ownerId)}`;
}

export function loadStudioContinuity(ownerId: string): StudioContinuity {
  const key = storageKey(ownerId);
  if (inMemory.has(key)) return inMemory.get(key)!;
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return {};
    const value = JSON.parse(raw) as StudioContinuity;
    if (!value || typeof value !== "object") return {};
    const result: StudioContinuity = {};
    if (typeof value.agentId === "string" && value.agentId.startsWith("agt_")) result.agentId = value.agentId;
    if (value.create && typeof value.create.key === "string" && value.create.key.startsWith("import-") && isDraft(value.create.draft)) {
      result.create = { key: value.create.key, draft: value.create.draft };
    }
    if (value.publish && typeof value.publish.key === "string" && value.publish.key.startsWith("publish-")
      && typeof value.publish.agentId === "string" && Number.isInteger(value.publish.draftVersion)
      && ["web", "telegram", "api"].includes(value.publish.channel)) result.publish = value.publish;
    inMemory.set(key, result);
    return result;
  } catch { return {}; }
}

export function saveStudioContinuity(ownerId: string, continuity: StudioContinuity): boolean {
  const key = storageKey(ownerId);
  inMemory.set(key, continuity);
  try { window.localStorage.setItem(key, JSON.stringify(continuity)); return true; }
  catch { return false; }
}

function isDraft(value: unknown): value is StudioDraft {
  if (!value || typeof value !== "object") return false;
  const draft = value as Partial<StudioDraft>;
  return ["concierge", "researcher", "developer"].includes(draft.templateId ?? "")
    && typeof draft.name === "string" && typeof draft.role === "string"
    && typeof draft.instructions === "string" && Array.isArray(draft.knowledge)
    && Array.isArray(draft.tools) && ["session", "approved", "continuous"].includes(draft.memoryMode ?? "")
    && ["web", "telegram", "api"].includes(draft.channel ?? "");
}
