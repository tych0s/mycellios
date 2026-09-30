import { mkdtempSync, readFileSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomBytes, randomUUID } from "node:crypto";
import Fastify from "fastify";
import { afterEach, describe, expect, it } from "vitest";
import { MeshDatabase } from "../src/storage/database.js";
import { registerLabUsbProvisioning, usbClaimHash } from "../scripts/lab-usb-provisioning.js";

const cleanup: (() => Promise<void>)[] = [];
afterEach(async () => { while (cleanup.length) await cleanup.pop()?.(); });
function setup(expired = false) {
  const root = mkdtempSync(join(tmpdir(), "mycellios-usb-"));
  const claimsPath = join(root, "claims.json");
  const token = randomBytes(32).toString("base64url");
  let time = Date.now();
  writeFileSync(claimsPath, '\uFEFF' + JSON.stringify([{ tokenHash: usbClaimHash(token), expiresAt: time + (expired ? -1 : 86_400_000) }]));
  const database = new MeshDatabase(join(root, "lab.db"));
  const app = Fastify();
  registerLabUsbProvisioning(app, { claimsPath, database, accountId: "test-owner", coordinatorUrl: "http://127.0.0.1:18790/", now: () => time });
  cleanup.push(async () => { await app.close(); database.close(); rmSync(root, { recursive: true, force: true }); });
  const post = (machineId: string, credential = token) => app.inject({ method: "POST", url: "/lab/usb/enrollment",
    headers: { "x-mycellios-install-claim": credential }, payload: { machineId } });
  return { post, database, claimsPath, advance: () => { time += 360_000; } };
}
describe("private USB enrollment claims", () => {
  it("denies missing, unknown and expired credentials without creating enrollments", async () => {
    const fixture = setup();
    expect((await fixture.post(randomUUID(), "wrong")).statusCode).toBe(403);
    expect((await fixture.post(randomUUID(), randomBytes(32).toString("base64url"))).statusCode).toBe(403);
    expect((await setup(true).post(randomUUID())).statusCode).toBe(403);
    expect(fixture.database.raw.prepare("SELECT count(*) AS n FROM node_enrollments").get()).toEqual({ n: 0 });
  });
  it("binds to one machine atomically, retries the same bundle and keeps only a token hash", async () => {
    const fixture = setup(); const machineId = randomUUID();
    const response = await fixture.post(machineId);
    expect(response.statusCode).toBe(200);
    expect(response.headers["cache-control"]).toBe("no-store");
    expect((await fixture.post(machineId)).json()).toEqual(response.json());
    const other = await Promise.all([fixture.post(randomUUID()), fixture.post(randomUUID())]);
    expect(other.map((entry) => entry.statusCode)).toEqual([403, 403]);
    expect(JSON.parse(readFileSync(fixture.claimsPath, "utf8"))[0].machineId).toBe(machineId);
  });
  it("refreshes an expiring bundle only for the bound machine and denies consumed enrollment", async () => {
    const fixture = setup(); const machineId = randomUUID();
    const first = (await fixture.post(machineId)).json(); fixture.advance();
    const fresh = (await fixture.post(machineId)).json();
    expect(fresh.enrollmentId).not.toBe(first.enrollmentId);
    const retired = fixture.database.raw.prepare("SELECT expires_at FROM node_enrollments WHERE id = ?")
      .get(first.enrollmentId) as { expires_at: number };
    expect(retired.expires_at).toBeLessThan(Date.parse(fresh.expiresAt));
    fixture.database.raw.prepare("UPDATE node_enrollments SET consumed_at = ? WHERE id = ?").run(Date.now(), fresh.enrollmentId);
    expect((await fixture.post(machineId)).statusCode).toBe(409);
    expect((await fixture.post(randomUUID())).statusCode).toBe(403);
  });
});
