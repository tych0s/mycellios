import { describe, expect, it } from "vitest";
import { SupabaseAuthService } from "../src/coordinator/supabase-auth.js";

describe("Supabase network authorization", () => {
  it("returns the authenticated user and network role", async () => {
    const calls: string[] = [];
    const auth = new SupabaseAuthService(
      "https://supabase.example.test",
      "service-role",
      async (input) => {
        const url = new URL(String(input));
        calls.push(url.pathname);
        if (url.pathname === "/auth/v1/user") {
          return Response.json({ id: "user-1", email: "owner@example.test" });
        }
        return Response.json([{ role: "owner" }]);
      },
    );

    await expect(auth.authenticate("user-access-token")).resolves.toEqual({
      id: "user-1",
      email: "owner@example.test",
      role: "owner",
    });
    expect(calls).toEqual(["/auth/v1/user", "/rest/v1/network_members"]);
  });

  it("rejects an expired access token without consulting membership", async () => {
    let calls = 0;
    const auth = new SupabaseAuthService(
      "https://supabase.example.test",
      "service-role",
      async () => {
        calls += 1;
        return new Response(null, { status: 401 });
      },
    );

    await expect(auth.authenticate("expired")).resolves.toBeNull();
    expect(calls).toBe(1);
  });
});
