import { describe, expect, it } from "vitest";
import { MeshDatabase } from "../src/storage/database.js";
import { StudioAgentError, StudioAgentStore } from "../src/coordinator/studio-agent-store.js";

const configuration = {
  name: "Mara", role: "Product guide",
  instructions: "Explain Mycellios accurately and state uncertainty whenever evidence is unavailable.",
  memoryMode: "approved" as const, knowledgeSourceIds: [], tools: ["documents" as const],
  modelPolicy: { preferredModel: "Qwen/Qwen3-0.6B", fallbackModel: null, privacy: "trusted-only" as const, maxOutputTokens: 512, deadlineMs: 120_000 },
};

function fixture() {
  const database = new MeshDatabase(":memory:");
  return { database, store: new StudioAgentStore(database) };
}

describe("StudioAgentStore", () => {
  it("creates idempotently and isolates owners", () => {
    const { database, store } = fixture();
    const created = store.create({ ownerId: "owner-a", idempotencyKey: "create-1", templateId: "concierge", configuration });
    expect(store.create({ ownerId: "owner-a", idempotencyKey: "create-1", templateId: "concierge", configuration }).id).toBe(created.id);
    expect(() => store.create({ ownerId: "owner-a", idempotencyKey: "create-1", templateId: "concierge", configuration: { ...configuration, name: "Different" } })).toThrow(/different draft/i);
    expect(store.list("owner-b")).toEqual([]);
    expect(store.get(created.id, "owner-b")).toBeNull();
    database.close();
  });

  it("uses optimistic draft versions", () => {
    const { database, store } = fixture();
    const created = store.create({ ownerId: "owner-a", idempotencyKey: "create-1", templateId: null, configuration });
    const updated = store.update(created.id, "owner-a", 1, { ...configuration, name: "Aster" });
    expect(updated.draftVersion).toBe(2);
    expect(() => store.update(created.id, "owner-a", 1, configuration)).toThrowError(StudioAgentError);
    database.close();
  });

  it("publishes an immutable revision while capacity is absent", () => {
    const { database, store } = fixture();
    const created = store.create({ ownerId: "owner-a", idempotencyKey: "create-1", templateId: "concierge", configuration });
    const published = store.publish({ agentId: created.id, ownerId: "owner-a", expectedVersion: 1, idempotencyKey: "publish-1", channels: ["web", "api"], hasCapacity: false });
    expect(published.agent.operationalState).toBe("waiting_for_capacity");
    expect(published.deployments.map((item) => item.state)).toEqual(["waiting_for_capacity", "waiting_for_capacity"]);
    expect(store.publish({ agentId: created.id, ownerId: "owner-a", expectedVersion: 1, idempotencyKey: "publish-1", channels: ["web", "api"], hasCapacity: true }).revision.id).toBe(published.revision.id);
    expect(() => store.publish({ agentId: created.id, ownerId: "owner-a", expectedVersion: 1, idempotencyKey: "publish-1", channels: ["web", "telegram"], hasCapacity: true })).toThrow(/different request/i);
    expect(() => database.raw.prepare("UPDATE studio_agent_revisions SET revision = 9").run()).toThrow(/immutable_studio_agent_revision/);
    database.close();
  });

  it("revokes channels when the agent is archived", () => {
    const { database, store } = fixture();
    const created = store.create({ ownerId: "owner-a", idempotencyKey: "create-1", templateId: null, configuration });
    store.publish({ agentId: created.id, ownerId: "owner-a", expectedVersion: 1, idempotencyKey: "publish-1", channels: ["web"], hasCapacity: false });
    expect(store.archive(created.id, "owner-a").operationalState).toBe("revoked");
    expect(store.deployments(created.id, "owner-a")[0]?.state).toBe("revoked");
    database.close();
  });

  it("rolls back by redeploying an exact immutable revision idempotently", () => {
    const { database, store } = fixture();
    const created = store.create({ ownerId: "owner-a", idempotencyKey: "create-1", templateId: null, configuration });
    const first = store.publish({ agentId: created.id, ownerId: "owner-a", expectedVersion: 1, idempotencyKey: "publish-1", channels: ["web"], hasCapacity: false });
    store.update(created.id, "owner-a", 1, { ...configuration, name: "Changed" });
    store.publish({ agentId: created.id, ownerId: "owner-a", expectedVersion: 2, idempotencyKey: "publish-2", channels: ["web"], hasCapacity: false });
    const rollback = store.rollback({ agentId: created.id, ownerId: "owner-a", revisionId: first.revision.id, idempotencyKey: "rollback-1", channels: ["web"], hasCapacity: false });
    expect(rollback.agent.publishedRevisionId).toBe(first.revision.id);
    expect(rollback.deployments[0]?.revisionId).toBe(first.revision.id);
    expect(store.rollback({ agentId: created.id, ownerId: "owner-a", revisionId: first.revision.id, idempotencyKey: "rollback-1", channels: ["web"], hasCapacity: true }).deployments[0]?.id).toBe(rollback.deployments[0]?.id);
    database.close();
  });
});
