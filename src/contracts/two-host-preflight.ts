import { z } from "zod";

export const TWO_HOST_PREFLIGHT_SCHEMA = "mycellios-two-host-preflight/1" as const;
export const twoHostPreflightCodeSchema = z.enum([
  "node_count_invalid", "node_identity_duplicate", "stage_endpoint_duplicate",
  "stage_endpoint_loopback", "stage_endpoint_placeholder", "remote_agent_missing",
  "remote_agent_insecure", "remote_auth_missing", "python_runtime_unavailable",
  "model_revision_unpinned", "stage_range_unproven", "transport_unmeasured",
]);
export const twoHostPreflightDiagnosticSchema = z.object({
  code: twoHostPreflightCodeSchema,
  severity: z.enum(["error", "warning"]),
  subject: z.string().min(1).max(128),
  message: z.string().min(1).max(512),
}).strict();
export const twoHostPreflightReportSchema = z.object({
  schema: z.literal(TWO_HOST_PREFLIGHT_SCHEMA),
  ready: z.boolean(),
  dryRun: z.boolean(),
  model: z.object({ source: z.string().min(1), revisionPinned: z.boolean() }).strict(),
  nodes: z.array(z.object({ id: z.string().min(1), remote: z.boolean(), endpointClass: z.enum(["routable", "loopback", "placeholder"]) }).strict()).min(2),
  capabilities: z.object({ python312: z.boolean(), remoteAuthentication: z.boolean(), measuredTransport: z.boolean(), preparedRanges: z.boolean() }).strict(),
  diagnostics: z.array(twoHostPreflightDiagnosticSchema),
}).strict().superRefine((report, context) => {
  const hasError = report.diagnostics.some((diagnostic) => diagnostic.severity === "error");
  if (report.ready === hasError) context.addIssue({ code: "custom", path: ["ready"], message: "Ready must equal absence of errors" });
});

export type TwoHostPreflightCode = z.infer<typeof twoHostPreflightCodeSchema>;
export type TwoHostPreflightDiagnostic = z.infer<typeof twoHostPreflightDiagnosticSchema>;
export type TwoHostPreflightReport = z.infer<typeof twoHostPreflightReportSchema>;
