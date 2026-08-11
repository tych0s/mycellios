import { z } from "zod";

export const NODE_IDENTITY_SCHEMA = "mycellios-node-identity/1" as const;

const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);

export const nodeIdentityKeySchema = z.object({
  channel: z.enum(["admission", "receipt"]),
  keyId: z.string().uuid(),
  algorithm: z.literal("ed25519"),
  publicKey: z.string().min(1).max(256),
  fingerprint: digestSchema,
  createdAt: z.string().datetime({ offset: true }),
  status: z.enum(["active", "revoked"]),
  revokedAt: z.string().datetime({ offset: true }).nullable(),
}).strict();

export const nodeIdentityDocumentSchema = z.object({
  schema: z.literal(NODE_IDENTITY_SCHEMA),
  version: z.literal(1),
  nodeId: z.string().uuid(),
  createdAt: z.string().datetime({ offset: true }),
  updatedAt: z.string().datetime({ offset: true }),
  keys: z.array(nodeIdentityKeySchema).length(2),
}).strict().superRefine((document, context) => {
  const channels = new Set(document.keys.map((key) => key.channel));
  if (channels.size !== 2) {
    context.addIssue({
      code: "custom",
      path: ["keys"],
      message: "node_identity_requires_separate_admission_and_receipt_keys",
    });
  }
});

export type NodeIdentityDocument = z.infer<typeof nodeIdentityDocumentSchema>;
export type NodeIdentityChannel = NodeIdentityDocument["keys"][number]["channel"];
