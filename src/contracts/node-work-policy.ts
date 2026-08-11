import { z } from "zod";

const scheduleWindowSchema = z.object({
  days: z.array(z.number().int().min(0).max(6)).min(1).max(7),
  startMinuteUtc: z.number().int().min(0).max(1_439),
  endMinuteUtc: z.number().int().min(1).max(1_440),
}).strict().superRefine((window, context) => {
  if (new Set(window.days).size !== window.days.length) context.addIssue({ code: "custom", path: ["days"], message: "days must be unique" });
  if (window.startMinuteUtc >= window.endMinuteUtc) context.addIssue({ code: "custom", path: ["endMinuteUtc"], message: "window must end after it starts" });
});

export const DEFAULT_NODE_WORK_POLICY: NodeWorkPolicy = {
  schedule: [{ days: [0, 1, 2, 3, 4, 5, 6], startMinuteUtc: 0, endMinuteUtc: 1_440 }],
  modelAllowlist: [],
};

export const nodeWorkPolicySchema = z.object({
  schedule: z.array(scheduleWindowSchema).max(32),
  /** Empty means every coordinator-certified model; non-empty is an exact allowlist. */
  modelAllowlist: z.array(z.string().min(1).max(256)).max(128),
}).strict().superRefine((policy, context) => {
  if (new Set(policy.modelAllowlist).size !== policy.modelAllowlist.length) {
    context.addIssue({ code: "custom", path: ["modelAllowlist"], message: "model allowlist must be unique" });
  }
});

export type NodeWorkPolicy = z.infer<typeof nodeWorkPolicySchema>;
export type NodeWorkPolicyRejection = "node_schedule_closed" | "node_model_not_allowed";

export function evaluateNodeWorkPolicy(
  input: unknown,
  model: string | null,
  at = new Date(),
): NodeWorkPolicyRejection | null {
  const policy = nodeWorkPolicySchema.parse(input);
  if (!Number.isFinite(at.getTime())) throw new Error("node_work_policy_time_is_invalid");
  const minute = at.getUTCHours() * 60 + at.getUTCMinutes();
  const day = at.getUTCDay();
  if (!policy.schedule.some((window) => window.days.includes(day) && minute >= window.startMinuteUtc && minute < window.endMinuteUtc)) {
    return "node_schedule_closed";
  }
  if (model !== null && policy.modelAllowlist.length > 0 && !policy.modelAllowlist.includes(model)) {
    return "node_model_not_allowed";
  }
  return null;
}
