import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, renameSync } from "node:fs";
import type { FastifyInstance } from "fastify";
import { z } from "zod";
import { NodeEnrollmentStore } from "../src/coordinator/node-enrollment-store.js";
import type { MeshDatabase } from "../src/storage/database.js";

const claimSchema = z.object({
  tokenHash: z.string().regex(/^[a-f0-9]{64}$/),
  expiresAt: z.number().int().positive(),
  machineId: z.string().optional(),
  enrollment: z.object({
    enrollmentId: z.string().uuid(), enrollmentToken: z.string(),
    nonce: z.string(), expiresAt: z.string(),
  }).optional(),
});
const claimsSchema = z.array(claimSchema);
const requestSchema = z.object({ machineId: z.string().uuid() }).strict();
export const usbClaimHash = (token: string): string => createHash("sha256").update(token).digest("hex");

/** Private lab route. The controller supplies independently issued USB claims. */
export function registerLabUsbProvisioning(app: FastifyInstance, options: {
  claimsPath: string; database: MeshDatabase; accountId: string;
  coordinatorUrl: string; now?: () => number;
}): void {
  const now = options.now ?? Date.now;
  const store = new NodeEnrollmentStore(options.database, now);
  app.post("/lab/usb/enrollment", { bodyLimit: 512 }, async (request, reply) => {
    const token = request.headers["x-mycellios-install-claim"];
    const body = requestSchema.safeParse(request.body);
    if (typeof token !== "string" || !/^[A-Za-z0-9_-]{43}$/.test(token) || !body.success) {
      return reply.code(403).send({ error: "installation_claim_denied" });
    }
    let claims: z.infer<typeof claimsSchema>;
    try { claims = claimsSchema.parse(JSON.parse(readFileSync(options.claimsPath, "utf8").replace(/^\uFEFF/, ""))); }
    catch { return reply.code(503).send({ error: "installation_claims_unavailable" }); }
    const claim = claims.find((entry) => entry.tokenHash === usbClaimHash(token));
    if (!claim || claim.expiresAt <= now() || (claim.machineId && claim.machineId !== body.data.machineId)) {
      return reply.code(403).send({ error: "installation_claim_denied" });
    }
    if (claim.enrollment) {
      const row = options.database.raw.prepare("SELECT consumed_at FROM node_enrollments WHERE id = ?")
        .get(claim.enrollment.enrollmentId) as { consumed_at: number | null } | undefined;
      if (row?.consumed_at !== null && row?.consumed_at !== undefined) {
        return reply.code(409).send({ error: "node_already_enrolled_retain_identity" });
      }
    }
    claim.machineId = body.data.machineId;
    if (!claim.enrollment || Date.parse(claim.enrollment.expiresAt) <= now() + 600_000) {
      if (claim.enrollment) {
        const retired = options.database.raw.prepare(
          "UPDATE node_enrollments SET expires_at = ? WHERE id = ? AND consumed_at IS NULL AND account_id = ?",
        ).run(now(), claim.enrollment.enrollmentId, options.accountId);
        if (retired.changes !== 1) return reply.code(409).send({ error: "node_enrollment_recovery_required" });
      }
      const enrollment = store.issue({ accountId: options.accountId,
        actor: { kind: "account", id: options.accountId, scopes: ["node:identity"] }, expiresInSeconds: 900 });
      store.confirm({ enrollmentId: enrollment.enrollmentId, accountId: options.accountId, actorId: options.accountId });
      claim.enrollment = enrollment;
    }
    // No await between lookup and atomic save: requests cannot steal a bound claim.
    const temporary = `${options.claimsPath}.tmp`;
    writeFileSync(temporary, `${JSON.stringify(claims)}\n`, { mode: 0o600 });
    renameSync(temporary, options.claimsPath);
    return reply.header("Cache-Control", "no-store").send({ schema: "mycellios-node-enrollment-bundle/1",
      coordinatorUrl: options.coordinatorUrl, ...claim.enrollment });
  });
}
