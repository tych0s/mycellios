import { z } from "zod";

const identifier = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/);
const nonSecretText = (maximum: number) => z.string().trim().min(1).max(maximum).refine(
  (value) => !/(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S+/i.test(value),
  "Studio configuration must not contain inline credentials",
);

export const studioMemoryModeSchema = z.enum(["session", "approved", "continuous"]);
export const studioToolIdSchema = z.enum(["documents", "calculator", "web", "api"]);
export const studioChannelSchema = z.enum(["web", "telegram", "api"]);
export const studioAgentStatusSchema = z.enum(["draft", "published", "archived"]);
export const studioOperationalStateSchema = z.enum([
  "draft",
  "validating",
  "waiting_for_capacity",
  "activating_model",
  "ready",
  "serving",
  "degraded",
  "unavailable",
  "revoked",
]);

export const studioAgentConfigurationSchema = z.object({
  name: nonSecretText(48),
  role: nonSecretText(100),
  instructions: nonSecretText(4_000),
  memoryMode: studioMemoryModeSchema,
  knowledgeSourceIds: z.array(identifier).max(64).default([]),
  tools: z.array(studioToolIdSchema).max(4).superRefine((tools, context) => {
    if (new Set(tools).size !== tools.length) context.addIssue({ code: "custom", message: "Studio tools must be unique" });
  }),
  modelPolicy: z.object({
    preferredModel: z.string().trim().min(1).max(256),
    fallbackModel: z.string().trim().min(1).max(256).nullable().default(null),
    privacy: z.enum(["default", "trusted-only"]).default("trusted-only"),
    maxOutputTokens: z.number().int().min(1).max(8_192).default(512),
    deadlineMs: z.number().int().min(1_000).max(300_000).default(120_000),
  }).strict(),
}).strict();

export const studioAgentCreateSchema = z.object({
  idempotencyKey: identifier,
  templateId: z.enum(["concierge", "researcher", "developer"]).nullable().default(null),
  configuration: studioAgentConfigurationSchema,
}).strict();

export const studioAgentUpdateSchema = z.object({
  expectedVersion: z.number().int().positive(),
  configuration: studioAgentConfigurationSchema,
}).strict();

export const studioPublishRequestSchema = z.object({
  idempotencyKey: identifier,
  expectedVersion: z.number().int().positive(),
  channels: z.array(studioChannelSchema).min(1).max(3).superRefine((channels, context) => {
    if (new Set(channels).size !== channels.length) context.addIssue({ code: "custom", message: "Studio channels must be unique" });
  }),
}).strict();

export const studioRollbackRequestSchema = z.object({
  idempotencyKey: identifier,
  revisionId: identifier,
  channels: z.array(studioChannelSchema).min(1).max(3),
}).strict();

export const studioKnowledgeIngestSchema = z.object({
  name: nonSecretText(200),
  mediaType: z.enum(["text/plain", "text/markdown"]),
  content: z.string().min(1).max(1_000_000),
}).strict();

export const studioMemoryFactCreateSchema = z.object({
  subjectId: identifier,
  fact: nonSecretText(1_000),
  origin: nonSecretText(200),
  confidence: z.number().min(0).max(1),
  approved: z.boolean().default(false),
  expiresAt: z.number().int().positive().nullable().optional(),
}).strict();

export const studioInvocationSchema = z.object({
  idempotencyKey: identifier,
  subjectId: identifier.default("anonymous"),
  message: nonSecretText(8_000),
  stream: z.boolean().default(false),
}).strict();

export interface StudioAgentConfiguration extends z.infer<typeof studioAgentConfigurationSchema> {}
export type StudioMemoryMode = z.infer<typeof studioMemoryModeSchema>;
export type StudioToolId = z.infer<typeof studioToolIdSchema>;
export type StudioChannel = z.infer<typeof studioChannelSchema>;
export type StudioAgentStatus = z.infer<typeof studioAgentStatusSchema>;
export type StudioOperationalState = z.infer<typeof studioOperationalStateSchema>;

export interface StudioAgentRecord {
  id: string;
  ownerId: string;
  templateId: "concierge" | "researcher" | "developer" | null;
  status: StudioAgentStatus;
  operationalState: StudioOperationalState;
  draftVersion: number;
  configuration: StudioAgentConfiguration;
  publishedRevisionId: string | null;
  createdAt: number;
  updatedAt: number;
  archivedAt: number | null;
}

export interface StudioAgentRevision {
  id: string;
  agentId: string;
  ownerId: string;
  revision: number;
  configuration: StudioAgentConfiguration;
  digest: string;
  createdAt: number;
}

export interface StudioChannelDeployment {
  id: string;
  agentId: string;
  revisionId: string;
  ownerId: string;
  channel: StudioChannel;
  state: "waiting_for_capacity" | "ready" | "degraded" | "revoked";
  publicId: string;
  createdAt: number;
  updatedAt: number;
  revokedAt: number | null;
}
