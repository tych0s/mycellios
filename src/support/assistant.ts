import { z } from "zod";
import type { ChatMessage } from "../contracts/types.js";

export const ASSISTANT_SETTINGS_ID = "landing-support";

export interface SupportAssistantSettings {
  enabled: boolean;
  modelId: string | null;
  systemPrompt: string;
  welcomeMessage: string;
  suggestions: string[];
  maxOutputTokens: number;
  temperature: number;
  allowDeviceControl: boolean;
  updatedAt: number;
}

export interface SupportAssistantNetworkContext {
  registeredNodes: number;
  connectedNodes: number;
  offeredMemoryGb: number;
  availableModels: string[];
  requestedModels: Array<{ id: string; status: string }>;
}

export const DEFAULT_SUPPORT_ASSISTANT_SETTINGS: SupportAssistantSettings = {
  enabled: true,
  modelId: null,
  systemPrompt: [
    "You are the official mycellios product and technical support assistant.",
    "Reply in the user's language, use plain language, and prefer short numbered steps.",
    "Ask for the operating system, GPU model, application version, and exact error before giving driver-specific instructions.",
    "Never invent a driver version, benchmark, connected node, model availability, download URL, or successful action.",
    "Use only the live network facts supplied in the system context.",
    "If an action changes device power or contribution settings, explain the effect and wait for the interface to request explicit confirmation.",
    "If the answer is uncertain, say so and guide the user to collect the missing evidence.",
  ].join("\n"),
  welcomeMessage: "Hi, I’m the mycellios assistant. I can explain the project, check the live network state, and help with installation, drivers, models, or device performance.",
  suggestions: [
    "How does mycellios work?",
    "I’m having trouble with my GPU drivers",
    "Which models are available right now?",
  ],
  maxOutputTokens: 512,
  temperature: 0.2,
  allowDeviceControl: true,
  updatedAt: 0,
};

export const supportAssistantSettingsUpdateSchema = z.object({
  enabled: z.boolean(),
  modelId: z.string().min(1).max(200).nullable(),
  systemPrompt: z.string().min(20).max(12_000),
  welcomeMessage: z.string().min(10).max(1_000),
  suggestions: z.array(z.string().min(3).max(180)).min(1).max(6),
  maxOutputTokens: z.number().int().min(64).max(4_096),
  temperature: z.number().min(0).max(1.5),
  allowDeviceControl: z.boolean(),
}).strict();

export const supportAssistantChatRequestSchema = z.object({
  session_id: z.string().min(1).max(160),
  messages: z.array(z.object({
    role: z.enum(["user", "assistant"]),
    content: z.string().min(1).max(4_000),
  }).strict()).min(1).max(20),
  page: z.string().min(1).max(240).optional(),
  platform: z.string().min(1).max(240).optional(),
}).strict().superRefine((value, context) => {
  const totalCharacters = value.messages.reduce((total, message) => total + message.content.length, 0);
  if (totalCharacters > 16_000) {
    context.addIssue({
      code: "custom",
      message: "The support conversation is too large.",
      path: ["messages"],
    });
  }
  if (value.messages.at(-1)?.role !== "user") {
    context.addIssue({
      code: "custom",
      message: "The last support message must come from the user.",
      path: ["messages"],
    });
  }
});

export function resolveSupportAssistantModel(
  settings: SupportAssistantSettings,
  availableModels: readonly string[],
): string | null {
  if (!settings.enabled) return null;
  if (settings.modelId) {
    return availableModels.includes(settings.modelId) ? settings.modelId : null;
  }
  return availableModels[0] ?? null;
}

export function buildSupportAssistantMessages(
  settings: SupportAssistantSettings,
  network: SupportAssistantNetworkContext,
  messages: Array<{ role: "user" | "assistant"; content: string }>,
  client: { page?: string; platform?: string },
): ChatMessage[] {
  const requestedModels = network.requestedModels.length > 0
    ? network.requestedModels.map((model) => `${model.id} (${model.status})`).join(", ")
    : "none";
  const availableModels = network.availableModels.length > 0
    ? network.availableModels.join(", ")
    : "none";
  const systemContext = [
    settings.systemPrompt,
    "",
    "Product facts:",
    "- mycellios coordinates real computers and compatible browser or desktop workers to run AI inference.",
    "- A connected device is not automatically an inference node: a compatible, verified runtime must announce a real model.",
    "- The network may use replicas or distributed stages; never claim a model is distributed unless the supplied live state proves it.",
    "- The desktop application can offer a chosen amount of memory and can use Automatic, GPU-only, or CPU-only compute mode.",
    "- Browser contribution uses WebGPU when the browser, operating system, and driver expose a compatible adapter.",
    "- Desktop inference runs through the installed Mycellios runtime; external model-serving daemons are not product execution paths.",
    "- The Tests area contains measured benchmark history. Do not turn configured or estimated throughput into a measurement.",
    "",
    "Live network state for this answer:",
    `- Registered nodes: ${network.registeredNodes}`,
    `- Connected nodes: ${network.connectedNodes}`,
    `- Offered memory: ${network.offeredMemoryGb.toFixed(1)} GB`,
    `- Available inference models: ${availableModels}`,
    `- Requested models: ${requestedModels}`,
    `- Current page: ${client.page ?? "unknown"}`,
    `- Client platform: ${client.platform ?? "unknown"}`,
    "",
    "Safety and actions:",
    "- You cannot silently change the network or the user's device.",
    "- The interface may offer a confirmation button for navigation or a local compute-mode change.",
    "- Shared model activation and administrator settings require an authorized administrator.",
    "- This response must be generated by the selected mycellios network model. Never suggest that an external AI was used.",
  ].join("\n");

  return [
    { role: "system", content: systemContext },
    ...messages.map((message) => ({ role: message.role, content: message.content })),
  ];
}
