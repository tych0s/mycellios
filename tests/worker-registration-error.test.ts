import { describe, expect, it } from "vitest";
import { readWorkerRegistrationError } from "../src/worker/registration-error.js";

describe("worker registration errors", () => {
  it("keeps the coordinator error code so cell enrollment can retry only reused keys", async () => {
    const response = new Response(JSON.stringify({
      error: { code: "worker_credential_reused", message: "Key belongs to another identity" },
    }), { status: 409 });
    const error = await readWorkerRegistrationError(response);
    expect(error).toMatchObject({ status: 409, code: "worker_credential_reused" });
  });

  it("keeps the status without trusting a non-JSON error body", async () => {
    const error = await readWorkerRegistrationError(new Response("Bad Gateway", { status: 502 }));
    expect(error).toMatchObject({ status: 502, code: null });
  });
});
