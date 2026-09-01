import { existsSync, readFileSync } from "node:fs";
import { z } from "zod";

const ids = ["legal_review", "accounting_review", "public_antifraud"] as const;
const policySchema = z.object({
  schema: z.literal("mycellios-economic-publication-policy/1"),
  asset: z.literal("MYC_MICROCREDITS"),
  internalLedgerOnly: z.boolean(),
  publicEarningsEnabled: z.boolean(),
  payoutEnabled: z.boolean(),
  reviewedAt: z.string().date(),
  controls: z.array(z.object({
    id: z.enum(ids), status: z.enum(["approved", "external-blocked", "residual-blocked"]),
    owner: z.string().min(1), prerequisite: z.string().min(1),
    evidence: z.array(z.string().min(1)).min(1), nextReviewAt: z.string().date(),
  }).strict()).length(ids.length),
}).strict().superRefine((policy, context) => {
  const unique = new Set(policy.controls.map(({ id }) => id));
  if (unique.size !== ids.length || ids.some((id) => !unique.has(id))) context.addIssue({ code: "custom", message: "economic publication controls are incomplete" });
  const approved = policy.controls.every(({ status }) => status === "approved");
  if ((policy.publicEarningsEnabled || policy.payoutEnabled || !policy.internalLedgerOnly) && !approved) {
    context.addIssue({ code: "custom", message: "public economics cannot open before every review is approved" });
  }
  if (policy.payoutEnabled && !policy.publicEarningsEnabled) context.addIssue({ code: "custom", message: "payout cannot precede public earnings approval" });
});

export function verifyEconomicPublicationPolicy(value: unknown, root = process.cwd()) {
  const policy = policySchema.parse(value);
  for (const control of policy.controls) for (const evidence of control.evidence) {
    if (!existsSync(`${root}/${evidence}`)) throw new Error(`economic_policy_evidence_missing:${control.id}:${evidence}`);
  }
  return policy;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const value = JSON.parse(readFileSync(process.argv[2] ?? "config/economic-publication-policy.json", "utf8"));
  const policy = verifyEconomicPublicationPolicy(value);
  process.stdout.write(`Economic publication policy verified: public earnings=${policy.publicEarningsEnabled}, payout=${policy.payoutEnabled}.\n`);
}
