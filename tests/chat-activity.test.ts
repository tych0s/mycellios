import { describe, expect, it } from "vitest";
import { CHAT_ACTIVITY_SCHEMA, chatActivitySchema } from "../src/contracts/chat-activity.js";

describe("chat activity metadata", () => {
  it("accepts a complete decision summary", () => {
    expect(chatActivitySchema.parse({ schema: CHAT_ACTIVITY_SCHEMA, model: "GPT-5.6 Sol", effort: "XHigh", rememberable: true, goal: { status: "needs_decision", title: "Choose the account usage behavior", tokens: 1590922, elapsedMs: 4980000 }, requestSummary: "Add account and usage details.", decision: { headline: "Backend data is required", pending: ["Connect usage history"], userAction: "Confirm the data source." } }).model).toBe("GPT-5.6 Sol");
  });

  it("rejects unknown fields and incomplete goals", () => {
    expect(() => chatActivitySchema.parse({ schema: CHAT_ACTIVITY_SCHEMA, model: "x", effort: null, rememberable: false, goal: null, requestSummary: null, decision: null, extra: true })).toThrow();
    expect(() => chatActivitySchema.parse({ schema: CHAT_ACTIVITY_SCHEMA, model: "x", effort: null, rememberable: false, goal: { status: "completed" }, requestSummary: null, decision: null })).toThrow();
  });
});
