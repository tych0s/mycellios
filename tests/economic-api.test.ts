import { afterEach, describe, expect, it } from "vitest";

import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";
import { verifyEconomicSettlementReceipt } from "../src/contracts/economic-settlement-receipt.js";
import { verifyExecutionReceipt } from "../src/contracts/execution-receipt.js";

const runtimes: CoordinatorRuntime[] = [];
afterEach(async () => Promise.all(runtimes.splice(0).map((runtime) => runtime.close())));
const sha = (character: string) => `sha256:${character.repeat(64)}`;

describe("economic API", () => {
  it("requires a durable receipt signer when the public API is exposed", async () => {
    await expect(createCoordinator({
      host: "0.0.0.0", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000,
      networkToken: "network-secret", apiAccessEnabled: true,
    }, { logger: false })).rejects.toThrow("receipt_signing_key_required_for_public_coordinator");
  });

  it("returns account-scoped usage economics and protects pricing mutations", async () => {
    const auth = new SupabaseAuthService("https://accounts.example.test", "service-role", (async (input) =>
      String(input).includes("/auth/v1/user")
        ? Response.json({ id: "user-economy", email: "owner@example.test" })
        : String(input).includes("/rest/v1/network_members")
          ? Response.json([{ role: "owner" }])
          : new Response(null, { status: 404 })) as typeof fetch);
    const runtime = await createCoordinator({
      host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000,
      apiAccessEnabled: true, apiStarterTokens: 10_000, modelAdminToken: "admin-secret",
    }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);
    const key = runtime.apiAccess.createKey("user-economy", "Economy test");
    const policy = runtime.economicLedger.resolvePricingPolicy("qwen3", "remote-replica", Date.now());
    expect(policy).not.toBeNull();
    const contributionEvidenceId = sha("b");
    const evidence = {
      schema: "mycellios-physical-contribution-evidence/1", id: contributionEvidenceId,
      jobId: "job-economy", executionReceiptId: sha("a"), traceDigest: sha("c"),
      participants: [{ nodeId: "node-economy", workerId: "worker-economy", physicalIdentityDigest: sha("d"), canaryEvidenceIds: [sha("e")], stageCount: 1 }],
      physicalBoundaryCount: 0, observedFrom: 1, observedUntil: 2, createdAt: 3,
    };
    runtime.database.raw.prepare("INSERT INTO economic_contribution_evidence(id, job_id, execution_receipt_id, trace_digest, evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?)")
      .run(contributionEvidenceId, "job-economy", sha("a"), sha("c"), JSON.stringify(evidence), 3);
    const settlement = runtime.economicLedger.settle({
      jobId: "job-economy", executionReceiptId: sha("a"), pricingPolicyId: policy!.id,
      contributionEvidenceId,
      payerAccountId: "user-economy", inputTokens: 100, outputTokens: 25,
      contributors: [{ nodeId: "node-economy", weight: 1 }], createdAt: Date.now(),
    });
    const response = await runtime.app.inject({
      method: "GET", url: "/v1/account/economy", headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(response.statusCode).toBe(200);
    expect(response.json()).toMatchObject({
      object: "economic_account", asset: "MYC_MICROCREDITS",
      balanceMicrounits: -200, settledJobs: 1,
    });
    const now = Date.now();
    runtime.database.raw.prepare(
      `INSERT INTO node_ownership(identity_kind, identity_id, account_id, credential_fingerprint,
       status, generation, created_at, updated_at) VALUES ('device', 'node-economy', 'user-economy', ?, 'active', 1, ?, ?)`,
    ).run(sha("f"), now, now);
    expect((await runtime.app.inject({ method: "GET", url: "/v1/nodes/node-economy/earnings",
      headers: { authorization: `Bearer ${key.secret}` } })).statusCode).toBe(403);
    const earnings = await runtime.app.inject({ method: "GET", url: "/v1/nodes/node-economy/earnings",
      headers: { authorization: "Bearer account-session" } });
    expect(earnings.json()).toMatchObject({ object: "node_earnings", asset_scope: "internal-ledger-only",
      public_earnings_enabled: false, payout_enabled: false, asset: "MYC_MICROCREDITS" });
    const receiptResponse = await runtime.app.inject({
      method: "GET", url: `/v1/economy/settlements/${encodeURIComponent(settlement.id)}/receipt`,
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(receiptResponse.statusCode).toBe(200);
    const keyResponse = await runtime.app.inject({
      method: "GET", url: "/v1/economy/receipt-key",
      headers: { authorization: `Bearer ${key.secret}` },
    });
    expect(keyResponse.statusCode).toBe(200);
    expect(verifyEconomicSettlementReceipt(receiptResponse.json(), keyResponse.json())).toEqual(receiptResponse.json());
    const otherKey = runtime.apiAccess.createKey("other-user", "Isolation test");
    expect((await runtime.app.inject({
      method: "GET", url: `/v1/economy/settlements/${encodeURIComponent(settlement.id)}/receipt`,
      headers: { authorization: `Bearer ${otherKey.secret}` },
    })).statusCode).toBe(404);
    runtime.store.createJob({ id: "job-receipt", sessionId: "session-receipt", model: "qwen3", workloadClass: "interactive", deadlineAt: Date.now() + 1_000 });
    const reservation = runtime.apiAccess.beginUsage("user-economy", key.id, {
      model: "qwen3", messages: [{ role: "user", content: "private prompt contents" }], max_tokens: 4,
    });
    runtime.apiAccess.attachJob(reservation.id, "job-receipt", "session-receipt");
    const executionReceipt = runtime.executionReceipts.record({
      schema: "mycellios-execution-receipt/1", jobId: "job-receipt", modelIdHash: sha("1"), routeClass: "replica",
      metrics: { inputTokens: 3, outputTokens: 1, ttftMs: 10, activeMs: 20 }, networkTraceDigest: sha("2"),
      recovery: { mode: "none", attempts: 1, replayedTokenEvents: 0 }, completedAt: Date.now(),
      privacy: { trust: "default", boundary: "trusted-edges", pinnedIdentityHashes: [] },
    });
    const topology = { schema: "mycellios-execution-topology/1", jobId: "job-receipt",
      receiptId: executionReceipt.receiptId, traceDigest: executionReceipt.networkTraceDigest,
      classification: "loopback", stages: [{ stageIndex: 0, alias: "stage-0", region: "Madrid", deviceType: "gpu", backend: "cuda" }],
      boundaries: [], createdAt: executionReceipt.completedAt };
    runtime.database.raw.prepare("INSERT INTO execution_topologies(job_id, receipt_id, trace_digest, topology_json, created_at) VALUES (?, ?, ?, ?, ?)")
      .run("job-receipt", executionReceipt.receiptId, executionReceipt.networkTraceDigest, JSON.stringify(topology), executionReceipt.completedAt);
    const executionResponse = await runtime.app.inject({ method: "GET", url: "/v1/requests/job-receipt/receipt",
      headers: { authorization: `Bearer ${key.secret}` } });
    const executionKeyResponse = await runtime.app.inject({ method: "GET", url: "/v1/execution-receipt-key",
      headers: { authorization: `Bearer ${key.secret}` } });
    expect(verifyExecutionReceipt(executionResponse.json(), executionKeyResponse.json())).toEqual(executionReceipt);
    expect(JSON.stringify(executionResponse.json())).not.toContain("private prompt contents");
    const topologyResponse = await runtime.app.inject({ method: "GET", url: "/v1/requests/job-receipt/topology",
      headers: { authorization: `Bearer ${key.secret}` } });
    expect(topologyResponse.statusCode).toBe(200);
    expect(topologyResponse.json()).toEqual(topology);
    const operations = await runtime.app.inject({ method: "GET", url: "/public/v1/admin/operations",
      headers: { authorization: "Bearer account-session" } });
    expect(operations.statusCode).toBe(200);
    expect(operations.json()).toMatchObject({ releases: { configured: false, items: [] },
      certifications: { configured: false, items: [], blocker: "durable_model_certification_registry_not_configured" },
      receipts: [{ receiptId: executionReceipt.receiptId, jobId: "job-receipt" }] });
    expect((await runtime.app.inject({ method: "GET", url: "/v1/requests/job-receipt/receipt",
      headers: { authorization: `Bearer ${otherKey.secret}` } })).statusCode).toBe(404);
    expect((await runtime.app.inject({ method: "GET", url: "/v1/requests/job-receipt/topology",
      headers: { authorization: `Bearer ${otherKey.secret}` } })).statusCode).toBe(404);
    const denied = await runtime.app.inject({
      method: "POST", url: "/internal/v1/economy/pricing", payload: {
        modelId: "qwen3", routeClass: "remote-replica", version: 2,
        inputMicrounitsPerToken: 2, outputMicrounitsPerToken: 8,
        platformFeeBps: 1_000, effectiveAt: 0, retiredAt: null,
      },
    });
    expect(denied.statusCode).toBe(401);
    const accepted = await runtime.app.inject({
      method: "POST", url: "/internal/v1/economy/pricing",
      headers: { "x-mycellios-admin-token": "admin-secret" },
      payload: {
        modelId: "qwen3", routeClass: "remote-replica", version: 2,
        inputMicrounitsPerToken: 2, outputMicrounitsPerToken: 8,
        platformFeeBps: 1_000, effectiveAt: 0, retiredAt: null,
      },
    });
    expect(accepted.statusCode).toBe(201);
  });
});
