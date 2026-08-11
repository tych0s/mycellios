import { describe, expect, it } from "vitest";
import { chatCompletionRequestSchema } from "../src/contracts/schemas.js";

const baseRequest = {
  model: "qwen",
  messages: [{ role: "user", content: "hello" }],
};

describe("inference privacy contract", () => {
  it("accepts explicit trusted-only and pinned boundary identities", () => {
    expect(chatCompletionRequestSchema.parse({
      ...baseRequest,
      privacy: {
        trust: "trusted-only",
        boundary: "pinned-edges",
        pinned_identity_ids: ["cell-owner", "cell-recipient"],
      },
    }).privacy).toEqual({
      trust: "trusted-only",
      boundary: "pinned-edges",
      pinned_identity_ids: ["cell-owner", "cell-recipient"],
    });
  });

  it("fails closed when pinned edges omit identities or IDs appear on an unpinned boundary", () => {
    expect(() => chatCompletionRequestSchema.parse({
      ...baseRequest,
      privacy: { trust: "default", boundary: "pinned-edges" },
    })).toThrow();
    expect(() => chatCompletionRequestSchema.parse({
      ...baseRequest,
      privacy: { trust: "default", boundary: "trusted-edges", pinned_identity_ids: ["cell-owner"] },
    })).toThrow();
  });
});
