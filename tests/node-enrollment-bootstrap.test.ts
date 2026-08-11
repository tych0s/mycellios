import { access, chmod, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { generateWorkerAdmissionCredential, workerAdmissionSigner } from "../src/worker/admission-credential.js";
import { NodeEnrollmentBootstrap } from "../src/node/enrollment-bootstrap.js";
import type { WorkerCapabilities } from "../src/contracts/types.js";

const cleanup: string[] = [];
afterEach(async () => Promise.all(cleanup.splice(0).map((path) => rm(path, { recursive: true, force: true }))));

describe("native node enrollment bootstrap", () => {
  it("redeems exact capabilities with a protected key and deletes the one-shot bundle", async () => {
    const root = await temporaryDirectory();
    const path = join(root, "enrollment.json");
    const bundle = enrollmentBundle();
    await writeFile(path, JSON.stringify(bundle), { mode: 0o600 });
    const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
    let body: Record<string, unknown> | null = null;
    const request = vi.fn(async (_url, init?: RequestInit) => {
      body = JSON.parse(String(init?.body));
      return Response.json({ enrollmentId: bundle.enrollmentId, accountId: "account-1",
        identity: { kind: "device", id: "node-1" }, credentialFingerprint: `sha256:${"a".repeat(64)}` });
    }) as typeof fetch;
    const bootstrap = new NodeEnrollmentBootstrap(path, "https://coordinator.example", request,
      () => Date.parse("2026-08-10T12:00:00Z"));
    await bootstrap.redeem({ identity: { kind: "device", id: "node-1" }, capabilities: capabilities(),
      protocol: { min: 1, max: 1 }, signer, registrationDigest: `sha256:${"b".repeat(64)}`, signal: new AbortController().signal });
    expect(body).toMatchObject({ schema: "mycellios-node-enrollment-redeem/1", enrollmentToken: bundle.enrollmentToken,
      nonce: bundle.nonce, identity: { kind: "device", id: "node-1" }, publicKey: signer.publicKey,
      registrationDigest: `sha256:${"b".repeat(64)}` });
    await expect(access(path)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("recovers the consume-before-delete crash window but retains retryable failures", async () => {
    const root = await temporaryDirectory();
    const consumedPath = join(root, "consumed.json");
    await writeFile(consumedPath, JSON.stringify(enrollmentBundle()), { mode: 0o600 });
    const consumed = new NodeEnrollmentBootstrap(consumedPath, "https://coordinator.example",
      vi.fn(async () => Response.json({ error: { code: "node_enrollment_already-consumed" } }, { status: 409 })) as typeof fetch);
    await consumed.redeem(input());
    await expect(access(consumedPath)).rejects.toMatchObject({ code: "ENOENT" });

    const retryPath = join(root, "retry.json");
    await writeFile(retryPath, JSON.stringify(enrollmentBundle()), { mode: 0o600 });
    const retry = new NodeEnrollmentBootstrap(retryPath, "https://coordinator.example",
      vi.fn(async () => Response.json({ error: { code: "node_enrollment_expired" } }, { status: 410 })) as typeof fetch);
    await expect(retry.redeem(input())).rejects.toThrow("node_enrollment_expired");
    await expect(readFile(retryPath, "utf8")).resolves.toContain("enrollmentToken");
  });

  it("rejects foreign coordinators, expired bundles and unsafe POSIX permissions before network access", async () => {
    const root = await temporaryDirectory();
    const request = vi.fn(async () => Response.json({}));
    const foreignPath = join(root, "foreign.json");
    await writeFile(foreignPath, JSON.stringify({ ...enrollmentBundle(), coordinatorUrl: "https://foreign.example" }), { mode: 0o600 });
    await expect(new NodeEnrollmentBootstrap(foreignPath, "https://coordinator.example", request as typeof fetch).redeem(input()))
      .rejects.toThrow("node_enrollment_bundle_coordinator_mismatch");
    const expiredPath = join(root, "expired.json");
    await writeFile(expiredPath, JSON.stringify({ ...enrollmentBundle(), expiresAt: "2020-01-01T00:00:00.000Z" }), { mode: 0o600 });
    await expect(new NodeEnrollmentBootstrap(expiredPath, "https://coordinator.example", request as typeof fetch).redeem(input()))
      .rejects.toThrow("node_enrollment_bundle_expired");
    if (process.platform !== "win32") {
      const unsafePath = join(root, "unsafe.json");
      await writeFile(unsafePath, JSON.stringify(enrollmentBundle()), { mode: 0o600 }); await chmod(unsafePath, 0o644);
      await expect(new NodeEnrollmentBootstrap(unsafePath, "https://coordinator.example", request as typeof fetch).redeem(input()))
        .rejects.toThrow("node_enrollment_bundle_permissions_are_unsafe");
    }
    expect(request).not.toHaveBeenCalled();
  });
});

function enrollmentBundle() {
  return { schema: "mycellios-node-enrollment-bundle/1", coordinatorUrl: "https://coordinator.example",
    enrollmentId: "4b5e61db-9e21-4268-b1a3-50dd0e818660", enrollmentToken: "t".repeat(43), nonce: "n".repeat(32),
    expiresAt: "2030-08-10T12:10:00.000Z" };
}

function input() {
  return { identity: { kind: "device" as const, id: "node-1" }, capabilities: capabilities(), protocol: { min: 1 as const, max: 1 as const },
    signer: workerAdmissionSigner(generateWorkerAdmissionCredential()), registrationDigest: `sha256:${"b".repeat(64)}`,
    signal: new AbortController().signal };
}

function capabilities(): WorkerCapabilities {
  return { region: "test", agentVersion: "test", gpus: [{ id: "cpu", vendor: "unknown", model: "CPU", physicalVramMb: 0,
    offeredVramMb: 512, freeOfferedVramMb: 512 }], deployments: [], limits: { maxConcurrency: 1, maxTemperatureC: 85,
    maxPowerW: 100, pauseWhenForeground: false }, network: { coordinatorRttMs: 1, uplinkMbps: 1, downlinkMbps: 1 } };
}

async function temporaryDirectory(): Promise<string> { const root = await mkdtemp(join(tmpdir(), "mycellios-enrollment-")); cleanup.push(root); return root; }
