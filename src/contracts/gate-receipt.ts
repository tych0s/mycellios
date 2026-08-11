import { z } from "zod";

export const GATE_RECEIPT_SCHEMA = "mycellios-gate-receipt/1" as const;

const digestSchema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const gitShaSchema = z.string().regex(/^[0-9a-f]{40}$/);

export const gateCheckSchema = z.object({
  id: z.string().min(1).max(128),
  kind: z.enum(["automatic", "physical", "external"]),
  status: z.enum(["pass", "fail", "blocked"]),
  detail: z.string().min(1).max(4_096),
  evidence: z.array(z.string().min(1).max(4_096)).max(128),
}).strict();

export const gateReceiptSchema = z.object({
  schema: z.literal(GATE_RECEIPT_SCHEMA),
  gate: z.string().regex(/^G[0-7]_[A-Z0-9_]+$/),
  phase: z.number().int().min(0).max(7),
  status: z.enum(["pass", "fail", "blocked", "obsolete"]),
  sourceSha: gitShaSchema,
  sourceIdentity: digestSchema,
  inputDigest: digestSchema,
  createdAt: z.string().datetime({ offset: true }),
  target: z.enum(["local", "pre", "release"]),
  checks: z.array(gateCheckSchema).min(1),
  hardware: z.array(z.object({
    id: z.string().min(1).max(256),
    class: z.string().min(1).max(128),
    fingerprint: digestSchema,
  }).strict()).max(128),
  blocker: z.string().min(1).max(4_096).nullable(),
}).strict().superRefine((receipt, context) => {
  const statuses = new Set(receipt.checks.map((check) => check.status));
  const expected = statuses.has("fail") ? "fail" : statuses.has("blocked") ? "blocked" : "pass";
  if (receipt.status !== "obsolete" && receipt.status !== expected) {
    context.addIssue({ code: "custom", path: ["status"], message: `Expected ${expected} from checks` });
  }
  if ((receipt.status === "blocked") !== (receipt.blocker !== null)) {
    context.addIssue({ code: "custom", path: ["blocker"], message: "Only blocked receipts require a blocker" });
  }
});

export type GateReceipt = z.infer<typeof gateReceiptSchema>;
