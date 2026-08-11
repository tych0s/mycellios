import { z } from "zod";
import { createPublicKey, type KeyLike } from "node:crypto";

import { sha256CanonicalEvidence } from "../core/json.js";
import {
  economicSettlementReceiptSchema,
  signEconomicSettlementReceipt,
  type EconomicSettlementReceipt,
} from "../contracts/economic-settlement-receipt.js";
import { MeshDatabase } from "../storage/database.js";
import { physicalContributionEvidenceSchema } from "./physical-contribution-evidence.js";

export const ECONOMIC_ASSET = "MYC_MICROCREDITS" as const;
const sha256Schema = z.string().regex(/^sha256:[0-9a-f]{64}$/);
const identifierSchema = z.string().min(1).max(256).regex(/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/);
const pricedModelSchema = z.union([identifierSchema, z.literal("*")]);
const nonNegativeSafeInteger = z.number().int().nonnegative().safe();

const pricingPolicyInputSchema = z.object({
  modelId: pricedModelSchema,
  routeClass: z.enum(["local-complete", "remote-replica", "distributed-pipeline"]),
  version: z.number().int().positive().safe(),
  inputMicrounitsPerToken: nonNegativeSafeInteger,
  outputMicrounitsPerToken: nonNegativeSafeInteger,
  platformFeeBps: z.number().int().min(0).max(10_000),
  effectiveAt: z.number().int().nonnegative().safe(),
  retiredAt: z.number().int().positive().safe().nullable(),
}).strict().refine(({ effectiveAt, retiredAt }) => retiredAt === null || retiredAt > effectiveAt, {
  message: "economic_pricing_window_is_invalid",
});

const settlementInputSchema = z.object({
  jobId: identifierSchema,
  executionReceiptId: sha256Schema,
  contributionEvidenceId: sha256Schema,
  pricingPolicyId: sha256Schema,
  executionRecovery: z.object({
    mode: z.enum(["none", "prompt-replay", "deterministic-prefix-replay"]),
    attempts: z.number().int().positive().safe(),
    replayedTokenEvents: z.number().int().nonnegative().safe(),
  }).strict().default({ mode: "none", attempts: 1, replayedTokenEvents: 0 }),
  payerAccountId: identifierSchema,
  inputTokens: nonNegativeSafeInteger,
  outputTokens: nonNegativeSafeInteger,
  contributors: z.array(z.object({
    nodeId: identifierSchema,
    weight: z.number().int().positive().safe(),
  }).strict()).min(1).max(1_024),
  createdAt: z.number().int().nonnegative().safe(),
}).strict().superRefine((input, context) => {
  if (new Set(input.contributors.map(({ nodeId }) => nodeId)).size !== input.contributors.length) {
    context.addIssue({ code: "custom", message: "economic_contributor_is_duplicated", path: ["contributors"] });
  }
});

export type PricingPolicyInput = z.infer<typeof pricingPolicyInputSchema>;
export type EconomicSettlementInput = z.infer<typeof settlementInputSchema>;

export interface PricingPolicy extends PricingPolicyInput { id: string }
export interface EconomicSettlement {
  id: string;
  jobId: string;
  executionReceiptId: string;
  contributionEvidenceId: string;
  pricingPolicyId: string;
  payerAccountId: string;
  grossMicrounits: number;
  providerMicrounits: number;
  platformMicrounits: number;
  contributorCredits: Array<{ nodeId: string; amountMicrounits: number }>;
  receipt: EconomicSettlementReceipt;
  createdAt: number;
}

export interface EconomicAccountSummary {
  asset: typeof ECONOMIC_ASSET;
  balanceMicrounits: number;
  settledJobs: number;
  lifetimeMicrounits: number;
}

export interface EconomicNodeReputation {
  verifiedJobs: number;
  verifiedStageCount: number;
  verifiedPhysicalBoundaries: number;
  earnedMicrounits: number;
  lastEvidenceId: string | null;
}

export class EconomicLedger {
  constructor(
    private readonly database: MeshDatabase,
    private readonly options: { receiptSigning: { keyId: string; privateKey: KeyLike } },
  ) {}

  registerPricingPolicy(value: unknown): PricingPolicy {
    const input = pricingPolicyInputSchema.parse(value);
    const id = sha256CanonicalEvidence({ schema: "mycellios-pricing-policy/1", ...input });
    this.database.transaction(() => {
      const existing = this.database.raw.prepare("SELECT * FROM pricing_policies WHERE id = ?").get(id);
      if (existing) return;
      const competing = this.database.raw.prepare(
        "SELECT id FROM pricing_policies WHERE model_id = ? AND route_class = ? AND version = ?",
      ).get(input.modelId, input.routeClass, input.version) as { id: string } | undefined;
      if (competing) throw new Error("economic_pricing_version_conflict");
      this.database.raw.prepare(`
        INSERT INTO pricing_policies(
          id, model_id, route_class, version, input_microunits_per_token,
          output_microunits_per_token, platform_fee_bps, effective_at, retired_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        id, input.modelId, input.routeClass, input.version,
        input.inputMicrounitsPerToken, input.outputMicrounitsPerToken,
        input.platformFeeBps, input.effectiveAt, input.retiredAt,
      );
    });
    return { id, ...input };
  }

  resolvePricingPolicy(modelId: string, routeClass: PricingPolicy["routeClass"], at: number): PricingPolicy | null {
    const row = this.database.raw.prepare(`
      SELECT id FROM pricing_policies
      WHERE model_id IN (?, '*') AND route_class = ? AND effective_at <= ?
        AND (retired_at IS NULL OR retired_at > ?)
      ORDER BY CASE WHEN model_id = ? THEN 0 ELSE 1 END, version DESC LIMIT 1
    `).get(identifierSchema.parse(modelId), routeClass, at, at, modelId) as { id: string } | undefined;
    return row ? this.readPricingPolicy(row.id) : null;
  }

  registerDefaultInternalPricing(): PricingPolicy[] {
    return (["remote-replica", "distributed-pipeline"] as const).map((routeClass) =>
      this.registerPricingPolicy({
        modelId: "*",
        routeClass,
        version: 1,
        inputMicrounitsPerToken: 1,
        outputMicrounitsPerToken: 4,
        platformFeeBps: 1_500,
        effectiveAt: 0,
        retiredAt: null,
      })
    );
  }

  receiptVerificationKey(): { keyId: string; spki: string } {
    const publicKey = createPublicKey(this.options.receiptSigning.privateKey);
    if (publicKey.asymmetricKeyType !== "ed25519") throw new Error("economic_receipt_public_key_is_not_ed25519");
    return {
      keyId: this.options.receiptSigning.keyId,
      spki: publicKey.export({ format: "der", type: "spki" }).toString("base64url"),
    };
  }

  settle(value: unknown): EconomicSettlement {
    const input = settlementInputSchema.parse(value);
    const requestDigest = sha256CanonicalEvidence({ schema: "mycellios-economic-settlement-request/1", ...input });
    return this.database.transaction(() => {
      const replay = this.database.raw.prepare(
        "SELECT id, request_digest FROM economic_settlements WHERE job_id = ?",
      ).get(input.jobId) as { id: string; request_digest: string } | undefined;
      if (replay) {
        if (replay.request_digest !== requestDigest) throw new Error("economic_settlement_job_conflict");
        return this.readSettlement(replay.id);
      }
      const receiptReplay = this.database.raw.prepare(
        "SELECT job_id FROM economic_settlements WHERE execution_receipt_id = ?",
      ).get(input.executionReceiptId) as { job_id: string } | undefined;
      if (receiptReplay) throw new Error("economic_settlement_receipt_replay");
      const policy = this.readPricingPolicy(input.pricingPolicyId);
      const evidenceRow = this.database.raw.prepare(
        "SELECT evidence_json FROM economic_contribution_evidence WHERE id = ?",
      ).get(input.contributionEvidenceId) as { evidence_json: string } | undefined;
      if (!evidenceRow) throw new Error("economic_contribution_evidence_not_found");
      const evidence = physicalContributionEvidenceSchema.parse(JSON.parse(evidenceRow.evidence_json));
      if (evidence.jobId !== input.jobId || evidence.executionReceiptId !== input.executionReceiptId) {
        throw new Error("economic_contribution_evidence_binding_is_invalid");
      }
      const verifiedNodes = evidence.participants.map(({ nodeId }) => nodeId).sort();
      const creditedNodes = input.contributors.map(({ nodeId }) => nodeId).sort();
      if (JSON.stringify(verifiedNodes) !== JSON.stringify(creditedNodes)) {
        throw new Error("economic_contributors_do_not_match_physical_evidence");
      }
      if (input.createdAt < policy.effectiveAt || (policy.retiredAt !== null && input.createdAt >= policy.retiredAt)) {
        throw new Error("economic_pricing_policy_is_not_effective");
      }
      const gross = safeBigIntToNumber(
        BigInt(input.inputTokens) * BigInt(policy.inputMicrounitsPerToken)
        + BigInt(input.outputTokens) * BigInt(policy.outputMicrounitsPerToken),
        "economic_settlement_amount_is_unsafe",
      );
      if (gross <= 0) throw new Error("economic_settlement_amount_is_zero");
      const provider = safeBigIntToNumber(
        BigInt(gross) * BigInt(10_000 - policy.platformFeeBps) / 10_000n,
        "economic_settlement_amount_is_unsafe",
      );
      const platform = gross - provider;
      const credits = distribute(provider, input.contributors);
      const settlementId = sha256CanonicalEvidence({ schema: "mycellios-economic-settlement/1", requestDigest, gross, provider, platform, credits });
      this.database.raw.prepare(`
        INSERT INTO economic_settlements(
          id, job_id, execution_receipt_id, contribution_evidence_id, pricing_policy_id, payer_account_id,
          input_tokens, output_tokens, gross_microunits, provider_microunits,
          platform_microunits, request_digest, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      `).run(
        settlementId, input.jobId, input.executionReceiptId, input.contributionEvidenceId, policy.id,
        input.payerAccountId, input.inputTokens, input.outputTokens, gross,
        provider, platform, requestDigest, input.createdAt,
      );
      const payer = this.ensureAccount("account", input.payerAccountId, input.createdAt);
      const platformAccount = this.ensureAccount("platform", "mycellios", input.createdAt);
      this.insertEntry(settlementId, payer, "usage", -gross, input.createdAt);
      for (const credit of credits) {
        if (credit.amountMicrounits === 0) continue;
        this.insertEntry(settlementId, this.ensureAccount("node", credit.nodeId, input.createdAt), "work", credit.amountMicrounits, input.createdAt);
      }
      if (platform > 0) this.insertEntry(settlementId, platformAccount, "platform_fee", platform, input.createdAt);
      this.assertBalanced(settlementId);
      for (const participant of evidence.participants) {
        const earned = credits.find(({ nodeId }) => nodeId === participant.nodeId)?.amountMicrounits ?? 0;
        this.database.raw.prepare(`
          INSERT INTO economic_node_reputation(
            node_id, verified_jobs, verified_stage_count, verified_physical_boundaries,
            earned_microunits, last_evidence_id, updated_at
          ) VALUES (?, 1, ?, ?, ?, ?, ?)
          ON CONFLICT(node_id) DO UPDATE SET
            verified_jobs = verified_jobs + 1,
            verified_stage_count = verified_stage_count + excluded.verified_stage_count,
            verified_physical_boundaries = verified_physical_boundaries + excluded.verified_physical_boundaries,
            earned_microunits = earned_microunits + excluded.earned_microunits,
            last_evidence_id = excluded.last_evidence_id,
            updated_at = excluded.updated_at
        `).run(
          participant.nodeId, participant.stageCount, evidence.physicalBoundaryCount,
          earned, evidence.id, input.createdAt,
        );
      }
      const receipt = signEconomicSettlementReceipt({
        schema: "mycellios-economic-settlement-receipt/1",
        settlementId,
        jobId: input.jobId,
        executionReceiptId: input.executionReceiptId,
        contributionEvidenceId: evidence.id,
        pricingPolicyId: policy.id,
        executionRecovery: input.executionRecovery,
        asset: ECONOMIC_ASSET,
        payerAccountHash: sha256CanonicalEvidence({ kind: "account", id: input.payerAccountId }),
        grossMicrounits: gross,
        providerMicrounits: provider,
        platformMicrounits: platform,
        contributorCredits: credits
          .map((credit) => ({
            nodeIdHash: sha256CanonicalEvidence({ kind: "node", id: credit.nodeId }),
            amountMicrounits: credit.amountMicrounits,
          }))
          .sort((left, right) => left.nodeIdHash.localeCompare(right.nodeIdHash)),
        createdAt: input.createdAt,
      }, this.options.receiptSigning);
      this.database.raw.prepare(`
        INSERT INTO economic_settlement_receipts(settlement_id, receipt_id, key_id, receipt_json, created_at)
        VALUES (?, ?, ?, ?, ?)
      `).run(settlementId, receipt.receiptId, receipt.keyId, JSON.stringify(receipt), input.createdAt);
      return this.readSettlement(settlementId);
    });
  }

  balance(ownerKind: "account" | "node" | "platform", ownerId: string): number {
    const row = this.database.raw.prepare(`
      SELECT COALESCE(SUM(entries.amount_microunits), 0) AS balance
      FROM economic_accounts accounts
      LEFT JOIN economic_ledger_entries entries ON entries.account_id = accounts.id
      WHERE accounts.owner_kind = ? AND accounts.owner_id = ? AND accounts.asset = ?
    `).get(ownerKind, identifierSchema.parse(ownerId), ECONOMIC_ASSET) as { balance: number };
    return Number(row.balance);
  }

  summary(ownerKind: "account" | "node" | "platform", ownerId: string): EconomicAccountSummary {
    const normalizedOwnerId = identifierSchema.parse(ownerId);
    const row = this.database.raw.prepare(`
      SELECT
        COALESCE(SUM(entries.amount_microunits), 0) AS balance,
        COUNT(DISTINCT entries.settlement_id) AS settled_jobs,
        COALESCE(SUM(CASE WHEN entries.amount_microunits > 0 THEN entries.amount_microunits ELSE 0 END), 0) AS lifetime
      FROM economic_accounts accounts
      LEFT JOIN economic_ledger_entries entries ON entries.account_id = accounts.id
      WHERE accounts.owner_kind = ? AND accounts.owner_id = ? AND accounts.asset = ?
    `).get(ownerKind, normalizedOwnerId, ECONOMIC_ASSET) as { balance: number; settled_jobs: number; lifetime: number };
    return {
      asset: ECONOMIC_ASSET,
      balanceMicrounits: Number(row.balance),
      settledJobs: Number(row.settled_jobs),
      lifetimeMicrounits: Number(row.lifetime),
    };
  }

  settlementReceiptForPayer(settlementId: string, payerAccountId: string): EconomicSettlementReceipt | null {
    const row = this.database.raw.prepare(`
      SELECT receipts.receipt_json
      FROM economic_settlements settlements
      JOIN economic_settlement_receipts receipts ON receipts.settlement_id = settlements.id
      WHERE settlements.id = ? AND settlements.payer_account_id = ?
    `).get(sha256Schema.parse(settlementId), identifierSchema.parse(payerAccountId)) as { receipt_json: string } | undefined;
    return row ? economicSettlementReceiptSchema.parse(JSON.parse(row.receipt_json)) : null;
  }

  nodeReputation(nodeId: string): EconomicNodeReputation {
    const row = this.database.raw.prepare("SELECT * FROM economic_node_reputation WHERE node_id = ?")
      .get(identifierSchema.parse(nodeId)) as Record<string, unknown> | undefined;
    return row ? {
      verifiedJobs: Number(row.verified_jobs),
      verifiedStageCount: Number(row.verified_stage_count),
      verifiedPhysicalBoundaries: Number(row.verified_physical_boundaries),
      earnedMicrounits: Number(row.earned_microunits),
      lastEvidenceId: String(row.last_evidence_id),
    } : {
      verifiedJobs: 0, verifiedStageCount: 0, verifiedPhysicalBoundaries: 0,
      earnedMicrounits: 0, lastEvidenceId: null,
    };
  }

  private readPricingPolicy(id: string): PricingPolicy {
    const row = this.database.raw.prepare("SELECT * FROM pricing_policies WHERE id = ?").get(sha256Schema.parse(id)) as Record<string, unknown> | undefined;
    if (!row) throw new Error("economic_pricing_policy_not_found");
    return {
      id: String(row.id), modelId: String(row.model_id), routeClass: String(row.route_class) as PricingPolicy["routeClass"],
      version: Number(row.version), inputMicrounitsPerToken: Number(row.input_microunits_per_token),
      outputMicrounitsPerToken: Number(row.output_microunits_per_token), platformFeeBps: Number(row.platform_fee_bps),
      effectiveAt: Number(row.effective_at), retiredAt: row.retired_at === null ? null : Number(row.retired_at),
    };
  }

  private readSettlement(id: string): EconomicSettlement {
    const row = this.database.raw.prepare("SELECT * FROM economic_settlements WHERE id = ?").get(id) as Record<string, unknown> | undefined;
    if (!row) throw new Error("economic_settlement_not_found");
    const credits = this.database.raw.prepare(`
      SELECT accounts.owner_id AS node_id, entries.amount_microunits AS amount
      FROM economic_ledger_entries entries JOIN economic_accounts accounts ON accounts.id = entries.account_id
      WHERE entries.settlement_id = ? AND entries.category = 'work' ORDER BY accounts.owner_id
    `).all(id) as Array<{ node_id: string; amount: number }>;
    const receiptRow = this.database.raw.prepare(
      "SELECT receipt_json FROM economic_settlement_receipts WHERE settlement_id = ?",
    ).get(id) as { receipt_json: string } | undefined;
    if (!receiptRow) throw new Error("economic_settlement_receipt_not_found");
    const receipt = economicSettlementReceiptSchema.parse(JSON.parse(receiptRow.receipt_json));
    return {
      id: String(row.id), jobId: String(row.job_id), executionReceiptId: String(row.execution_receipt_id),
      contributionEvidenceId: String(row.contribution_evidence_id),
      pricingPolicyId: String(row.pricing_policy_id), payerAccountId: String(row.payer_account_id),
      grossMicrounits: Number(row.gross_microunits), providerMicrounits: Number(row.provider_microunits),
      platformMicrounits: Number(row.platform_microunits),
      contributorCredits: credits.map(({ node_id, amount }) => ({ nodeId: node_id, amountMicrounits: Number(amount) })),
      receipt,
      createdAt: Number(row.created_at),
    };
  }

  private ensureAccount(ownerKind: "account" | "node" | "platform", ownerId: string, createdAt: number): string {
    const id = sha256CanonicalEvidence({ schema: "mycellios-economic-account/1", ownerKind, ownerId, asset: ECONOMIC_ASSET });
    this.database.raw.prepare("INSERT OR IGNORE INTO economic_accounts(id, owner_kind, owner_id, asset, created_at) VALUES (?, ?, ?, ?, ?)")
      .run(id, ownerKind, ownerId, ECONOMIC_ASSET, createdAt);
    return id;
  }

  private insertEntry(settlementId: string, accountId: string, category: "usage" | "work" | "platform_fee", amount: number, createdAt: number): void {
    const id = sha256CanonicalEvidence({ schema: "mycellios-economic-ledger-entry/1", settlementId, accountId, category, amount });
    this.database.raw.prepare("INSERT INTO economic_ledger_entries(id, settlement_id, account_id, category, amount_microunits, created_at) VALUES (?, ?, ?, ?, ?, ?)")
      .run(id, settlementId, accountId, category, amount, createdAt);
  }

  private assertBalanced(settlementId: string): void {
    const row = this.database.raw.prepare("SELECT COALESCE(SUM(amount_microunits), 0) AS total FROM economic_ledger_entries WHERE settlement_id = ?")
      .get(settlementId) as { total: number };
    if (Number(row.total) !== 0) throw new Error("economic_ledger_is_unbalanced");
  }
}

function distribute(total: number, contributors: EconomicSettlementInput["contributors"]): Array<{ nodeId: string; amountMicrounits: number }> {
  const ordered = [...contributors].sort((left, right) => left.nodeId.localeCompare(right.nodeId));
  const totalWeight = ordered.reduce((sum, contributor) => sum + BigInt(contributor.weight), 0n);
  const credits = ordered.map((contributor) => ({
    nodeId: contributor.nodeId,
    amountMicrounits: safeBigIntToNumber(BigInt(total) * BigInt(contributor.weight) / totalWeight, "economic_settlement_amount_is_unsafe"),
  }));
  let remainder = total - credits.reduce((sum, credit) => sum + credit.amountMicrounits, 0);
  for (const credit of credits) {
    if (remainder === 0) break;
    credit.amountMicrounits += 1;
    remainder -= 1;
  }
  return credits;
}

function safeBigIntToNumber(value: bigint, error: string): number {
  if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(error);
  return Number(value);
}
