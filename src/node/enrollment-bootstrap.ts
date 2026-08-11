import { lstat, readFile, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { z } from "zod";
import { nodeEnrollmentBundleSchema, nodeEnrollmentRedeemSchema } from "../contracts/node-control.js";
import type { WorkerCapabilities } from "../contracts/types.js";
import type { WorkerAdmissionSigner } from "../worker/admission-credential.js";

const responseSchema = z.object({
  enrollmentId: z.string().uuid(),
  accountId: z.string().min(1),
  identity: z.object({ kind: z.enum(["device", "cell"]), id: z.string().min(1) }).strict(),
  credentialFingerprint: z.string().regex(/^sha256:[a-f0-9]{64}$/),
}).strict();

export class NodeEnrollmentBootstrap {
  readonly path: string;
  constructor(
    path: string,
    private readonly expectedCoordinatorUrl: string,
    private readonly request: typeof fetch = fetch,
    private readonly now: () => number = Date.now,
  ) { this.path = resolve(path); }

  async redeem(input: {
    identity: { kind: "device" | "cell"; id: string };
    capabilities: WorkerCapabilities;
    protocol: { min: 1; max: 1 };
    signer: WorkerAdmissionSigner;
    registrationDigest: string;
    signal: AbortSignal;
  }): Promise<void> {
    let stats;
    try { stats = await lstat(this.path); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return; throw error; }
    if (!stats.isFile() || stats.isSymbolicLink()) throw new Error("node_enrollment_bundle_is_not_a_regular_file");
    if (process.platform !== "win32" && (stats.mode & 0o077) !== 0) throw new Error("node_enrollment_bundle_permissions_are_unsafe");
    if (stats.size > 4_096) throw new Error("node_enrollment_bundle_is_too_large");
    const bundle = nodeEnrollmentBundleSchema.parse(JSON.parse(await readFile(this.path, "utf8")));
    if (Date.parse(bundle.expiresAt) <= this.now()) throw new Error("node_enrollment_bundle_expired");
    const expected = normalizedCoordinator(this.expectedCoordinatorUrl);
    if (normalizedCoordinator(bundle.coordinatorUrl) !== expected) throw new Error("node_enrollment_bundle_coordinator_mismatch");
    const body = nodeEnrollmentRedeemSchema.parse({
      schema: "mycellios-node-enrollment-redeem/1",
      enrollmentToken: bundle.enrollmentToken,
      identity: input.identity,
      publicKey: input.signer.publicKey,
      protocol: input.protocol,
      registrationDigest: input.registrationDigest,
      nonce: bundle.nonce,
    });
    const endpoint = new URL("/internal/v1/nodes/enrollments/redeem", expected);
    const response = await this.request(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.any([input.signal, AbortSignal.timeout(10_000)]),
      redirect: "manual",
    });
    const text = await response.text();
    if (Buffer.byteLength(text, "utf8") > 16 * 1024) throw new Error("node_enrollment_response_is_too_large");
    const payload = parseJson(text);
    if (!response.ok) {
      const code = errorCode(payload);
      if (response.status === 409 && code === "node_enrollment_already-consumed") {
        await rm(this.path, { force: true });
        return;
      }
      throw new Error(code ?? `node_enrollment_http_${response.status}`);
    }
    const result = responseSchema.parse(payload);
    if (result.enrollmentId !== bundle.enrollmentId || result.identity.kind !== input.identity.kind || result.identity.id !== input.identity.id) {
      throw new Error("node_enrollment_response_identity_mismatch");
    }
    await rm(this.path, { force: true });
  }
}

function normalizedCoordinator(value: string): string {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash) throw new Error("node_enrollment_coordinator_url_is_unsafe");
  const loopback = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]).has(url.hostname.toLowerCase());
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) throw new Error("node_enrollment_coordinator_transport_is_unsafe");
  url.pathname = "/"; url.search = ""; url.hash = "";
  return url.toString();
}

function parseJson(value: string): unknown {
  try { return JSON.parse(value) as unknown; } catch { throw new Error("node_enrollment_response_is_invalid"); }
}

function errorCode(value: unknown): string | null {
  const parsed = z.object({ error: z.object({ code: z.string().regex(/^[A-Za-z0-9._:-]{1,128}$/) }).passthrough() }).passthrough().safeParse(value);
  return parsed.success ? parsed.data.error.code : null;
}
