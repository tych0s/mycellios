export type StudioTemplateId = "concierge" | "researcher" | "developer";
export type MemoryMode = "session" | "approved" | "continuous";
export type StudioToolId = "web" | "documents" | "calculator" | "api";
export type StudioChannelId = "web" | "telegram" | "api";

export interface StudioDraft {
  templateId: StudioTemplateId;
  name: string;
  role: string;
  instructions: string;
  memoryMode: MemoryMode;
  knowledge: string[];
  tools: StudioToolId[];
  channel: StudioChannelId;
}

export interface StudioTemplate {
  id: StudioTemplateId;
  eyebrow: string;
  title: string;
  description: string;
  draft: StudioDraft;
}

export const STUDIO_STORAGE_KEY = "mycellios.studio.draft.v1";

export const STUDIO_TEMPLATES: readonly StudioTemplate[] = [
  {
    id: "concierge",
    eyebrow: "Customer-facing",
    title: "Product concierge",
    description: "Explains a product, qualifies intent, and hands complex cases to a person.",
    draft: {
      templateId: "concierge",
      name: "Mara",
      role: "Mycellios product guide",
      instructions: "Be warm, concise, and concrete. Explain the product in plain language, ask one useful follow-up at a time, and never invent availability or pricing.",
      memoryMode: "approved",
      knowledge: ["Product handbook", "Pricing and access"],
      tools: ["documents", "calculator"],
      channel: "web",
    },
  },
  {
    id: "researcher",
    eyebrow: "Knowledge work",
    title: "Research companion",
    description: "Synthesizes trusted sources and keeps decisions traceable across sessions.",
    draft: {
      templateId: "researcher",
      name: "Aster",
      role: "Evidence-first research companion",
      instructions: "Separate evidence from inference. Cite the supplied knowledge, surface uncertainty, and finish with the most useful next question or decision.",
      memoryMode: "continuous",
      knowledge: ["Research library", "Decision log"],
      tools: ["web", "documents", "calculator"],
      channel: "web",
    },
  },
  {
    id: "developer",
    eyebrow: "Developer tool",
    title: "API copilot",
    description: "Helps teams integrate an API without turning Studio into a general-purpose IDE.",
    draft: {
      templateId: "developer",
      name: "Nori",
      role: "Mycellios integration copilot",
      instructions: "Answer with the smallest correct integration path. Prefer typed examples, state assumptions, and never request or repeat credentials.",
      memoryMode: "session",
      knowledge: ["API reference", "Integration playbook"],
      tools: ["documents", "api"],
      channel: "api",
    },
  },
] as const;

const templateIds = new Set(STUDIO_TEMPLATES.map((template) => template.id));
const memoryModes = new Set<MemoryMode>(["session", "approved", "continuous"]);
const toolIds = new Set<StudioToolId>(["web", "documents", "calculator", "api"]);
const channelIds = new Set<StudioChannelId>(["web", "telegram", "api"]);

export function draftFromTemplate(id: StudioTemplateId): StudioDraft {
  const source = STUDIO_TEMPLATES.find((template) => template.id === id) ?? STUDIO_TEMPLATES[0]!;
  return { ...source.draft, knowledge: [...source.draft.knowledge], tools: [...source.draft.tools] };
}

export function restoreStudioDraft(raw: string | null): StudioDraft {
  if (!raw) return draftFromTemplate("concierge");
  try {
    const value = JSON.parse(raw) as Partial<StudioDraft>;
    const templateId = templateIds.has(value.templateId as StudioTemplateId)
      ? value.templateId as StudioTemplateId
      : "concierge";
    const fallback = draftFromTemplate(templateId);
    return {
      templateId,
      name: cleanText(value.name, fallback.name, 48),
      role: cleanText(value.role, fallback.role, 100),
      instructions: cleanText(value.instructions, fallback.instructions, 600),
      memoryMode: memoryModes.has(value.memoryMode as MemoryMode) ? value.memoryMode as MemoryMode : fallback.memoryMode,
      knowledge: cleanStringList(value.knowledge, 6, 64),
      tools: cleanEnumList(value.tools, toolIds, 4),
      channel: channelIds.has(value.channel as StudioChannelId) ? value.channel as StudioChannelId : fallback.channel,
    };
  } catch {
    return draftFromTemplate("concierge");
  }
}

export function studioCompletion(draft: StudioDraft): number {
  const checks = [
    draft.name.trim().length > 1,
    draft.role.trim().length > 3,
    draft.instructions.trim().length > 20,
    draft.knowledge.length > 0,
    draft.tools.length > 0,
  ];
  return Math.round((checks.filter(Boolean).length / checks.length) * 100);
}

export function previewReply(draft: StudioDraft, prompt: string): string {
  const question = prompt.trim();
  if (!question) return `Ask ${draft.name || "this identity"} something to test its voice, knowledge, and boundaries.`;
  const knowledge = draft.knowledge.length > 0
    ? `I would ground this in ${draft.knowledge.slice(0, 2).join(" and ")}`
    : "I do not have an approved knowledge source for this yet";
  const toolNote = draft.tools.includes("web")
    ? " and verify anything time-sensitive before answering."
    : ".";
  return `${knowledge}${toolNote} As ${draft.role || "your configured assistant"}, I would answer “${question}” in the voice and boundaries defined here.`;
}

function cleanText(value: unknown, fallback: string, max: number): string {
  return typeof value === "string" && value.trim() ? value.trim().slice(0, max) : fallback;
}

function cleanStringList(value: unknown, maxItems: number, maxLength: number): string[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => item.trim().slice(0, maxLength))
    .filter(Boolean)
    .slice(0, maxItems);
}

function cleanEnumList<T extends string>(value: unknown, allowed: Set<T>, maxItems: number): T[] {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter((item): item is T => typeof item === "string" && allowed.has(item as T)))].slice(0, maxItems);
}
