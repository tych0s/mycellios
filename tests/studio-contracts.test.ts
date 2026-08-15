import { describe, expect, it } from "vitest";
import { studioAgentCreateSchema, studioAgentUpdateSchema, studioPublishRequestSchema } from "../src/contracts/studio.js";

const configuration = {
  name: "Mara",
  role: "Product guide",
  instructions: "Explain Mycellios accurately and state uncertainty when evidence is unavailable.",
  memoryMode: "approved",
  knowledgeSourceIds: [],
  tools: ["documents"],
  modelPolicy: { preferredModel: "Qwen/Qwen3-0.6B", fallbackModel: null, privacy: "trusted-only", maxOutputTokens: 512, deadlineMs: 120_000 },
} as const;

describe("Studio contracts", () => {
  it("accepts a bounded agent draft", () => {
    expect(studioAgentCreateSchema.parse({ idempotencyKey: "create-1", templateId: "concierge", configuration })).toMatchObject({ templateId: "concierge" });
  });

  it("rejects inline credentials and duplicate capabilities", () => {
    expect(() => studioAgentUpdateSchema.parse({ expectedVersion: 1, configuration: { ...configuration, instructions: "api_key=do-not-store" } })).toThrow();
    expect(() => studioAgentUpdateSchema.parse({ expectedVersion: 1, configuration: { ...configuration, tools: ["documents", "documents"] } })).toThrow();
  });

  it("requires optimistic concurrency and unique publication channels", () => {
    expect(() => studioAgentUpdateSchema.parse({ expectedVersion: 0, configuration })).toThrow();
    expect(() => studioPublishRequestSchema.parse({ idempotencyKey: "publish-1", expectedVersion: 1, channels: ["web", "web"] })).toThrow();
  });
});
