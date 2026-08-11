import { describe, expect, it } from "vitest";
import { formatRecoveryMode } from "./Panel";

describe("inference recovery presentation", () => {
  it("labels replay honestly instead of calling it a checkpoint resume", () => {
    expect(formatRecoveryMode("deterministic-prefix-replay")).toBe("Exact prefix replay");
    expect(formatRecoveryMode("prompt-replay")).toBe("Prompt replay");
  });
});
