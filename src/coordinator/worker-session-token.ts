import { createHmac, timingSafeEqual } from "node:crypto";
import { z } from "zod";

export const WORKER_SESSION_TOKEN_PREFIX = "mws1";

const workerSessionClaimsSchema = z.object({
  schema: z.literal("mycellios-worker-session/1"),
  workerId: z.string().min(1).max(256),
  identityKind: z.enum(["device", "cell"]),
  identityId: z.string().min(1).max(256),
  credentialFingerprint: z.string().regex(/^sha256:[a-f0-9]{64}$/),
}).strict();

export type WorkerSessionClaims = z.infer<typeof workerSessionClaimsSchema>;

export function issueWorkerSessionToken(
  secret: string,
  claims: Omit<WorkerSessionClaims, "schema">,
): string {
  const payload = Buffer.from(JSON.stringify({
    schema: "mycellios-worker-session/1",
    ...claims,
  } satisfies WorkerSessionClaims), "utf8").toString("base64url");
  return `${WORKER_SESSION_TOKEN_PREFIX}.${payload}.${signWorkerSessionPayload(secret, payload)}`;
}

export function verifyWorkerSessionToken(
  secret: string,
  token: string,
): WorkerSessionClaims | null {
  const [prefix, payload, signature, extra] = token.split(".");
  if (
    prefix !== WORKER_SESSION_TOKEN_PREFIX
    || !payload
    || !signature
    || extra !== undefined
    || !/^[A-Za-z0-9_-]+$/.test(payload)
    || !/^[A-Za-z0-9_-]+$/.test(signature)
  ) return null;
  const expected = Buffer.from(signWorkerSessionPayload(secret, payload), "utf8");
  const received = Buffer.from(signature, "utf8");
  if (expected.length !== received.length || !timingSafeEqual(expected, received)) return null;
  try {
    return workerSessionClaimsSchema.parse(
      JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as unknown,
    );
  } catch {
    return null;
  }
}

function signWorkerSessionPayload(secret: string, payload: string): string {
  return createHmac("sha256", secret).update(payload, "utf8").digest("base64url");
}
