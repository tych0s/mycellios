import { z } from "zod";

export const economicReadinessGateIdSchema = z.enum([
  "legal_opinion",
  "operating_jurisdiction",
  "tax_reporting",
  "kyc_aml",
  "sanctions_screening",
  "fraud_controls",
  "payment_provider_contract",
  "custody_model",
  "accounting_reconciliation",
  "security_audit",
  "incident_response",
  "public_token_necessity",
  "securities_analysis",
  "market_abuse_controls",
  "token_governance",
  "smart_contract_audit",
  "liquidity_risk",
]);

export const economicReadinessEvidenceSchema = z.object({
  gate: economicReadinessGateIdSchema,
  status: z.enum(["verified", "failed"]),
  evidenceRef: z.string().min(1).max(512),
  verifiedAt: z.string().datetime(),
  expiresAt: z.string().datetime().nullable(),
}).strict();

export const economicReadinessReceiptSchema = z.object({
  sequence: z.number().int().positive(),
  receiptDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  previousReceiptDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/).nullable(),
  evidenceDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  gate: economicReadinessGateIdSchema,
  status: z.enum(["verified", "failed"]),
  recordedAt: z.string().datetime(),
}).strict();

export const economicReadinessAssessmentSchema = z.object({
  object: z.literal("economic_readiness"),
  schemaVersion: z.literal("economic-readiness-v1"),
  evaluatedAt: z.string().datetime(),
  evidenceReceipts: z.array(economicReadinessReceiptSchema).max(25),
  internalCredits: z.object({
    enabled: z.literal(true),
    currency: z.literal("compute_credit"),
    nonMonetary: z.literal(true),
    transferable: z.literal(false),
    withdrawable: z.literal(false),
  }).strict(),
  payments: z.object({
    state: z.enum(["blocked", "ready_for_sandbox", "ready_for_live"]),
    enabled: z.boolean(),
    provider: z.null(),
    missingGates: z.array(economicReadinessGateIdSchema),
    failedGates: z.array(economicReadinessGateIdSchema),
  }).strict(),
  publicToken: z.object({
    decision: z.enum(["no_go", "eligible_for_spec"]),
    enabled: z.literal(false),
    network: z.null(),
    missingGates: z.array(economicReadinessGateIdSchema),
    failedGates: z.array(economicReadinessGateIdSchema),
  }).strict(),
}).strict();

export type EconomicReadinessEvidence = z.infer<typeof economicReadinessEvidenceSchema>;
export type EconomicReadinessReceipt = z.infer<typeof economicReadinessReceiptSchema>;
export type EconomicReadinessAssessment = z.infer<typeof economicReadinessAssessmentSchema>;
