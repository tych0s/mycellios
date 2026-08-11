import { describe, expect, it } from "vitest";
import { evaluateNodeWorkPolicy, nodeWorkPolicySchema } from "../src/contracts/node-work-policy.js";

const policy = {
  schedule: [{ days: [1, 2, 3, 4, 5], startMinuteUtc: 8 * 60, endMinuteUtc: 18 * 60 }],
  modelAllowlist: ["org/approved-model"],
};

describe("node work policy", () => {
  it("admits only exact models inside an explicit UTC window", () => {
    expect(evaluateNodeWorkPolicy(policy, "org/approved-model", new Date("2026-08-10T12:00:00Z"))).toBeNull();
    expect(evaluateNodeWorkPolicy(policy, "org/other-model", new Date("2026-08-10T12:00:00Z"))).toBe("node_model_not_allowed");
    expect(evaluateNodeWorkPolicy(policy, "org/approved-model", new Date("2026-08-10T20:00:00Z"))).toBe("node_schedule_closed");
  });

  it("fails closed for empty schedules and rejects ambiguous policy input", () => {
    expect(evaluateNodeWorkPolicy({ schedule: [], modelAllowlist: [] }, null, new Date("2026-08-10T12:00:00Z"))).toBe("node_schedule_closed");
    expect(nodeWorkPolicySchema.safeParse({ schedule: [{ days: [1, 1], startMinuteUtc: 100, endMinuteUtc: 90 }], modelAllowlist: ["x", "x"] }).success).toBe(false);
  });
});
