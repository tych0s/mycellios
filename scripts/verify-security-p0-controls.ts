import { existsSync, readFileSync } from "node:fs";
import { z } from "zod";

const controlIds = [
  "os_isolation",
  "account_rate_limits",
  "append_only_audit",
  "owner_key_recovery",
  "production_tls_perimeter",
  "sybil_abuse_resistance",
  "independent_crypto_review",
] as const;

const controlSchema = z.object({
  id: z.enum(controlIds),
  status: z.enum(["implemented", "external-blocked", "residual-blocked"]),
  owner: z.string().min(1),
  evidence: z.array(z.string().min(1)).min(1),
  residual: z.string().min(1),
  prerequisite: z.string().min(1).optional(),
  nextReviewAt: z.string().date(),
}).strict().superRefine((control, context) => {
  if (control.status !== "implemented" && !control.prerequisite) {
    context.addIssue({ code: "custom", message: "blocked security controls require a prerequisite" });
  }
});

const matrixSchema = z.object({
  schema: z.literal("mycellios-security-p0-controls/1"),
  publicNetworkAllowed: z.boolean(),
  reviewedAt: z.string().date(),
  controls: z.array(controlSchema).length(controlIds.length),
}).strict().superRefine((matrix, context) => {
  const ids = new Set(matrix.controls.map(({ id }) => id));
  for (const id of controlIds) if (!ids.has(id)) context.addIssue({ code: "custom", message: `missing security control: ${id}` });
  if (ids.size !== matrix.controls.length) context.addIssue({ code: "custom", message: "duplicate security control" });
  if (matrix.publicNetworkAllowed && matrix.controls.some(({ status }) => status !== "implemented")) {
    context.addIssue({ code: "custom", message: "public network cannot open with blocked security controls" });
  }
});

export function verifySecurityP0Controls(value: unknown, root = process.cwd()) {
  const matrix = matrixSchema.parse(value);
  for (const control of matrix.controls) {
    for (const evidence of control.evidence) {
      if (!existsSync(`${root}/${evidence}`)) throw new Error(`security_p0_evidence_missing:${control.id}:${evidence}`);
    }
  }
  return matrix;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const path = process.argv[2] ?? "config/security-p0-controls.json";
  const matrix = verifySecurityP0Controls(JSON.parse(readFileSync(path, "utf8")));
  const implemented = matrix.controls.filter(({ status }) => status === "implemented").length;
  const blocked = matrix.controls.length - implemented;
  process.stdout.write(`Security P0 controls verified: ${implemented} implemented, ${blocked} blocked with owners; public network allowed=${matrix.publicNetworkAllowed}.\n`);
}
