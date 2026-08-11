import { describe, expect, it } from "vitest";

import {
  encodeDevelopmentLabInvitation,
  parseDevelopmentLabInvitation,
} from "../src/contracts/development-lab.js";

const invitation = {
  schema: "mycellios-development-lab-invitation/1",
  coordinatorUrl: "https://www.mycellios.com",
  labId: "lab_abcdefghijklmnopqrstuv",
  token: Buffer.alloc(32, 0xa5).toString("base64url"),
  expiresAt: "2026-08-01T08:00:00.000Z",
} as const;

describe("development lab invitation codec", () => {
  it("round-trips one canonical pasteable code without leaking fields around it", () => {
    const code = encodeDevelopmentLabInvitation(invitation);
    expect(code).toMatch(/^mycellios-dev-lab:[A-Za-z0-9_-]+$/);
    expect(code).not.toContain(invitation.token);
    expect(parseDevelopmentLabInvitation(code)).toEqual(invitation);
  });

  it("requires HTTPS except for an explicitly allowed loopback test coordinator", () => {
    expect(() => encodeDevelopmentLabInvitation({
      ...invitation,
      coordinatorUrl: "http://updates.example.test",
    })).toThrow("coordinator_url_invalid");
    expect(() => encodeDevelopmentLabInvitation({
      ...invitation,
      coordinatorUrl: "http://127.0.0.1:8787",
    })).toThrow("coordinator_url_invalid");
    const loopback = encodeDevelopmentLabInvitation({
      ...invitation,
      coordinatorUrl: "http://127.0.0.1:8787",
    }, { allowLoopbackHttp: true });
    expect(parseDevelopmentLabInvitation(
      loopback,
      { allowLoopbackHttp: true },
    ).coordinatorUrl).toBe("http://127.0.0.1:8787");
  });

  it("rejects non-canonical or oversized invitation text", () => {
    const code = encodeDevelopmentLabInvitation(invitation);
    expect(() => parseDevelopmentLabInvitation(`${code}=`)).toThrow(
      "invitation_invalid",
    );
    expect(() => parseDevelopmentLabInvitation(
      `mycellios-dev-lab:${"a".repeat(5_000)}`,
    )).toThrow("invitation_invalid");
  });
});
