import { z } from "zod";

export const CHAT_ACTIVITY_SCHEMA = "mycellios-chat-activity/1" as const;

export const chatActivitySchema = z.object({
  schema: z.literal(CHAT_ACTIVITY_SCHEMA),
  model: z.string().min(1).max(160),
  effort: z.string().min(1).max(80).nullable(),
  rememberable: z.boolean(),
  goal: z.object({
    status: z.enum(["in_progress", "completed", "blocked", "needs_decision"]),
    title: z.string().min(1).max(2_000),
    tokens: z.number().int().nonnegative().nullable(),
    elapsedMs: z.number().int().nonnegative().nullable(),
  }).strict().nullable(),
  requestSummary: z.string().min(1).max(2_000).nullable(),
  decision: z.object({
    headline: z.string().min(1).max(500),
    pending: z.array(z.string().min(1).max(1_000)).max(20),
    userAction: z.string().min(1).max(1_000).nullable(),
  }).strict().nullable(),
}).strict();

export type ChatActivity = z.infer<typeof chatActivitySchema>;
