import { generateKeyPairSync } from "node:crypto";
import { afterEach, describe, expect, it } from "vitest";
import { buildModelCertification, signModelCertification } from "../src/contracts/model-certification.js";
import { createCoordinator, type CoordinatorRuntime } from "../src/coordinator/server.js";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

const runtimes: CoordinatorRuntime[] = [];
afterEach(async () => Promise.all(runtimes.splice(0).map((runtime) => runtime.close())));
const sha = (value: string) => `sha256:${value.repeat(64)}` as const;

describe("operator evidence API", () => {
  it("publishes a pinned certification and exposes its redacted operator summary", async () => {
    const keys = generateKeyPairSync("ed25519");
    const pinned = { keyId: "cert-key", spki: keys.publicKey.export({ format: "der", type: "spki" }).toString("base64url") };
    let requestedRole: "operator" | "admin" = "admin";
    const auth = new SupabaseAuthService("https://accounts.example.test", "service-role", (async (input, init) => {
      if (String(input).includes("/auth/v1/user")) {
        const token = String((init?.headers as Record<string, string> | undefined)?.authorization ?? "");
        requestedRole = token.includes("operator") ? "operator" : "admin";
        return Response.json({ id: `${requestedRole}-user`, email: "ops@example.test" });
      }
      if (String(input).includes("/rest/v1/network_members")) {
        return Response.json([{ role: requestedRole }]);
      }
      return new Response(null, { status: 404 });
    }) as typeof fetch);
    const runtime = await createCoordinator({ host: "127.0.0.1", port: 0, databasePath: ":memory:", requestTimeoutMs: 1_000,
      apiAccessEnabled: true, modelCertificationPinnedKeys: [pinned] }, { logger: false, supabaseAuthService: auth });
    runtimes.push(runtime);
    const certification = signModelCertification(buildModelCertification({ decision: "certified", modelFamily: "qwen3-dense",
      distributionManifestId: sha("a"), adapterContractId: sha("b"), componentManifestId: sha("c"),
      topology: { kind: "local-complete", stageCount: 1, tensorParallelDegree: 1 }, hardware: { platform: "linux", arch: "x64",
        backend: "cuda", deviceFamily: "sm89", minimumMemoryBytes: 8_000_000_000, driverFingerprint: sha("d") },
      context: { maximumTokens: 8192, codecs: ["fp16/1"], workerProtocol: { min: 8, max: 8 }, tensorAbi: "tensor/1" },
      evidence: { class: "physical", receiptId: sha("e"), sourceRevision: "f".repeat(40), measuredAt: "2026-08-10T00:00:00.000Z" },
      review: { reviewerId: "reviewer", reviewedAt: "2026-08-10T01:00:00.000Z" }, expiresAt: "2027-08-10T00:00:00.000Z" }),
    { keyId: pinned.keyId, privateKey: keys.privateKey });
    const published = await runtime.app.inject({ method: "PUT", url: "/public/v1/admin/model-certifications",
      headers: { authorization: "Bearer admin-session" }, payload: certification });
    expect(published.statusCode).toBe(201);
    const preflight = await runtime.app.inject({ method: "OPTIONS", url: "/public/v1/admin/operations" });
    expect(preflight.statusCode).toBe(204);
    expect(preflight.headers["access-control-allow-origin"]).toBe("*");
    const read = await runtime.app.inject({ method: "GET", url: "/public/v1/admin/operations",
      headers: { authorization: "Bearer operator-session" } });
    expect(read.statusCode).toBe(200);
    expect(read.json().certifications).toMatchObject({ configured: true, items: [{ certificationId: certification.certificationId,
      modelFamily: "qwen3-dense", decision: "certified" }], blocker: null });
  });
});
