import { z } from "zod";
import {
  workerAdmissionIdentitySchema,
  workerAdmissionProofSchema,
  workerAdmissionPublicKeySchema,
  workerProtocolRangeSchema,
} from "./worker-admission.js";
import { DEFAULT_NODE_WORK_POLICY, nodeWorkPolicySchema } from "./node-work-policy.js";

export const NODE_CONTROL_PROTOCOL_VERSION = 1 as const;
export const NODE_CONTROL_MAX_CLOCK_SKEW_MS = 30_000;
export const NODE_EVENT_ORIGIN_CURSOR = "evt_0_0000000000000000" as const;

const identifier = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/);
const runtimeIdentifier = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$/);
const sha256 = z.string().regex(/^sha256:[a-f0-9]{64}$/);
const nonce = z.string().min(22).max(128).regex(/^[A-Za-z0-9_-]+$/);
const timestamp = z.string().datetime({ offset: true });
const generation = z.number().int().positive().max(Number.MAX_SAFE_INTEGER);
const cursor = z.string().regex(/^evt_[0-9]{1,20}_[a-f0-9]{16}$/);

export const nodeControlActorSchema = z.object({
  kind: z.enum(["account", "operator", "node"]),
  id: identifier,
  scopes: z.array(z.enum([
    "node:read",
    "node:control",
    "node:limits",
    "node:update",
    "node:identity",
  ])).min(1).max(5),
}).strict().superRefine((actor, context) => {
  if (new Set(actor.scopes).size !== actor.scopes.length) {
    context.addIssue({ code: "custom", path: ["scopes"], message: "scopes must be unique" });
  }
});

export const nodeEnrollmentCreateSchema = z.object({
  schema: z.literal("mycellios-node-enrollment-create/1"),
  accountId: identifier,
  requestedBy: nodeControlActorSchema,
  expiresInSeconds: z.number().int().min(60).max(900),
}).strict();

export const nodeEnrollmentRedeemSchema = z.object({
  schema: z.literal("mycellios-node-enrollment-redeem/1"),
  enrollmentToken: z.string().min(32).max(256).regex(/^[A-Za-z0-9_-]+$/),
  identity: workerAdmissionIdentitySchema,
  publicKey: workerAdmissionPublicKeySchema,
  protocol: workerProtocolRangeSchema,
  registrationDigest: sha256,
  nonce,
}).strict();

export const nodeEnrollmentBundleSchema = z.object({
  schema: z.literal("mycellios-node-enrollment-bundle/1"),
  coordinatorUrl: z.string().url(),
  enrollmentId: z.string().uuid(),
  enrollmentToken: z.string().min(32).max(256).regex(/^[A-Za-z0-9_-]+$/),
  nonce,
  expiresAt: timestamp,
}).strict();

export const nodeProofOfPossessionSchema = z.object({
  schema: z.literal("mycellios-node-proof-of-possession/1"),
  identity: workerAdmissionIdentitySchema,
  proof: workerAdmissionProofSchema,
  issuedAt: timestamp,
  expiresAt: timestamp,
  nonce,
}).strict();

const commandBase = {
  schema: z.literal("mycellios-node-command/1"),
  id: z.string().uuid(),
  nodeId: identifier,
  actor: nodeControlActorSchema,
  generation,
  issuedAt: timestamp,
  expiresAt: timestamp,
  nonce,
} as const;

const emptyPayload = z.object({ version: z.literal(1) }).strict();
const limitsPayload = z.object({
  version: z.literal(1),
  maxConcurrency: z.number().int().min(1).max(64),
  maxCpuPercent: z.number().int().min(1).max(100),
  maxRamMiB: z.number().int().min(256),
  maxVramMiB: z.number().int().min(0),
  maxDiskMiB: z.number().int().min(512),
  maxTemperatureC: z.number().int().min(40).max(110),
}).strict();

export const nodeCommandSchema = z.discriminatedUnion("type", [
  z.object({ ...commandBase, type: z.literal("pause"), payload: emptyPayload }).strict(),
  z.object({ ...commandBase, type: z.literal("resume"), payload: emptyPayload }).strict(),
  z.object({ ...commandBase, type: z.literal("drain"), payload: z.object({ version: z.literal(1), deadlineMs: z.number().int().min(1_000).max(3_600_000) }).strict() }).strict(),
  z.object({ ...commandBase, type: z.literal("set-limits"), payload: limitsPayload }).strict(),
  z.object({ ...commandBase, type: z.literal("set-policy"), payload: z.object({ version: z.literal(1), policy: nodeWorkPolicySchema }).strict() }).strict(),
  z.object({ ...commandBase, type: z.literal("update"), payload: z.object({ version: z.literal(1), channel: z.enum(["dev", "stable"]), manifestId: sha256.optional() }).strict() }).strict(),
  z.object({ ...commandBase, type: z.literal("rollback"), payload: z.object({ version: z.literal(1), componentIds: z.array(identifier).min(1).max(16) }).strict() }).strict(),
  z.object({ ...commandBase, type: z.literal("revoke"), payload: z.object({ version: z.literal(1), reason: z.string().min(1).max(256) }).strict() }).strict(),
  z.object({ ...commandBase, type: z.literal("uninstall"), payload: z.object({ version: z.literal(1), retain: z.object({ cache: z.boolean(), logs: z.boolean(), configuration: z.boolean(), identity: z.boolean() }).strict() }).strict() }).strict(),
]);

export const nodeCommandResultSchema = z.object({
  schema: z.literal("mycellios-node-command-result/1"),
  id: z.string().uuid(),
  commandId: z.string().uuid(),
  nodeId: identifier,
  generation,
  state: z.enum(["accepted", "applied", "rejected", "expired"]),
  observedAt: timestamp,
  resultDigest: sha256,
  error: z.object({ code: identifier, message: z.string().min(1).max(512) }).strict().nullable(),
}).strict();

export const nodeDesiredStateSchema = z.object({
  schema: z.literal("mycellios-node-desired-state/1"),
  nodeId: identifier,
  generation,
  contributionEnabled: z.boolean(),
  drain: z.boolean(),
  updateChannel: z.enum(["dev", "stable"]),
  limits: limitsPayload.omit({ version: true }),
  policy: nodeWorkPolicySchema.optional(),
  updatedAt: timestamp,
  updatedBy: nodeControlActorSchema,
}).strict();

export const nodeSnapshotSchema = z.object({
  schema: z.literal("mycellios-node-snapshot/1"),
  nodeId: identifier,
  generation,
  cursor,
  observedAt: timestamp,
  state: z.enum(["loading", "pairing", "canary", "ready", "reconnecting", "degraded", "failed", "revoked"]),
  contributionEnabled: z.boolean(),
  draining: z.boolean(),
  updateChannel: z.enum(["dev", "stable"]),
  limits: limitsPayload.omit({ version: true }),
  policy: nodeWorkPolicySchema.optional(),
  activeCommandIds: z.array(z.string().uuid()).max(256),
  build: z.object({ version: z.string().min(1).max(128), sourceRevision: z.string().regex(/^[a-f0-9]{40}$/) }).strict(),
  runtime: z.object({ ready: z.boolean(), abi: runtimeIdentifier.nullable(), backend: runtimeIdentifier.nullable(), uninstallAvailable: z.boolean().optional() }).strict(),
  diagnostics: z.object({
    capturedAt: timestamp,
    configRevision: z.number().int().positive(),
    capacity: z.object({
      acceleratorCount: z.number().int().nonnegative(),
      primaryAccelerator: z.object({ name: z.string().min(1).max(200), totalMemoryMiB: z.number().int().nonnegative(), offeredMemoryMiB: z.number().int().nonnegative() }).strict().nullable(),
    }).strict(),
    resources: z.object({
      cpuPercent: z.number().min(0).max(100),
      ramMiB: z.number().nonnegative(),
      diskMiB: z.number().nonnegative(),
      temperatureC: z.number().nullable(),
      healthy: z.boolean(),
      violation: z.string().regex(/^[a-zA-Z0-9_.:-]{1,120}$/).nullable(),
    }).strict().nullable(),
    incident: z.object({ code: z.string().regex(/^[a-zA-Z0-9_.:-]{1,120}$/), occurredAt: timestamp }).strict().nullable(),
  }).strict().optional(),
}).strict();

export const nodeEventSchema = z.object({
  schema: z.literal("mycellios-node-event/1"),
  cursor,
  nodeId: identifier,
  generation,
  actor: nodeControlActorSchema,
  type: identifier,
  occurredAt: timestamp,
  payloadDigest: sha256,
  previousEventDigest: sha256.nullable(),
}).strict();

const commandScopes: Record<z.infer<typeof nodeCommandSchema>["type"], string> = {
  pause: "node:control",
  resume: "node:control",
  drain: "node:control",
  "set-limits": "node:limits",
  "set-policy": "node:limits",
  update: "node:update",
  rollback: "node:update",
  revoke: "node:identity",
  uninstall: "node:identity",
};

export interface NodeCommandValidationContext {
  now?: Date;
  minimumGeneration: number;
  consumedNonces: ReadonlySet<string>;
  expectedNodeId: string;
}

export function parseAuthorizedNodeCommand(
  input: unknown,
  context: NodeCommandValidationContext,
): z.infer<typeof nodeCommandSchema> {
  const command = nodeCommandSchema.parse(input);
  const now = (context.now ?? new Date()).getTime();
  if (command.nodeId !== context.expectedNodeId) throw new Error("node_command_wrong_node");
  if (command.generation < context.minimumGeneration) throw new Error("node_command_generation_downgrade");
  if (context.consumedNonces.has(command.nonce)) throw new Error("node_command_replay");
  if (Date.parse(command.issuedAt) > now + NODE_CONTROL_MAX_CLOCK_SKEW_MS) throw new Error("node_command_issued_in_future");
  if (Date.parse(command.expiresAt) <= now) throw new Error("node_command_expired");
  if (Date.parse(command.expiresAt) <= Date.parse(command.issuedAt)) throw new Error("node_command_invalid_ttl");
  if (!command.actor.scopes.includes(commandScopes[command.type] as never)) throw new Error("node_command_scope_denied");
  if (command.actor.kind === "node" && command.type !== "rollback") throw new Error("node_command_actor_denied");
  return command;
}

export const nodeControlOpenApiSchemas = Object.freeze({
  NodeEnrollmentCreate: z.toJSONSchema(nodeEnrollmentCreateSchema, { io: "input" }),
  NodeEnrollmentRedeem: z.toJSONSchema(nodeEnrollmentRedeemSchema, { io: "input" }),
  NodeProofOfPossession: z.toJSONSchema(nodeProofOfPossessionSchema, { io: "input" }),
  NodeCommand: z.toJSONSchema(nodeCommandSchema, { io: "input" }),
  NodeCommandResult: z.toJSONSchema(nodeCommandResultSchema, { io: "input" }),
  NodeDesiredState: z.toJSONSchema(nodeDesiredStateSchema, { io: "input" }),
  NodeSnapshot: z.toJSONSchema(nodeSnapshotSchema, { io: "input" }),
  NodeEvent: z.toJSONSchema(nodeEventSchema, { io: "input" }),
});

export type NodeCommand = z.infer<typeof nodeCommandSchema>;
export type NodeCommandResult = z.infer<typeof nodeCommandResultSchema>;
export type NodeDesiredState = z.infer<typeof nodeDesiredStateSchema>;
export type NodeSnapshot = z.infer<typeof nodeSnapshotSchema>;
export type NodeEvent = z.infer<typeof nodeEventSchema>;
export type NodeEnrollmentBundle = z.infer<typeof nodeEnrollmentBundleSchema>;
