import { z } from "zod";

export const NODE_LIFECYCLE_STATE_SCHEMA = "mycellios-node-lifecycle-state/1" as const;

export const nodeLifecycleStateSchema = z.object({
  schema: z.literal(NODE_LIFECYCLE_STATE_SCHEMA),
  version: z.literal(1),
  generation: z.number().int().nonnegative(),
  status: z.enum(["starting", "running", "draining", "stopped", "failed"]),
  instanceId: z.string().uuid().nullable(),
  updatedAt: z.string().datetime({ offset: true }),
  lastStartedAt: z.string().datetime({ offset: true }).nullable(),
  lastStoppedAt: z.string().datetime({ offset: true }).nullable(),
  failureCode: z.string().regex(/^[a-z0-9_:-]+$/).max(128).nullable(),
}).strict();

export type NodeLifecycleState = z.infer<typeof nodeLifecycleStateSchema>;
