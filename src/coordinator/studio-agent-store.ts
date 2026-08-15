import { createHash, randomUUID } from "node:crypto";
import type { MeshDatabase } from "../storage/database.js";
import {
  studioAgentConfigurationSchema,
  type StudioAgentConfiguration,
  type StudioAgentRecord,
  type StudioAgentRevision,
  type StudioChannel,
  type StudioChannelDeployment,
} from "../contracts/studio.js";

export class StudioAgentError extends Error {
  constructor(readonly code: string, message: string, readonly statusCode = 400) {
    super(message);
  }
}

export class StudioAgentStore {
  constructor(private readonly database: MeshDatabase) {}

  create(input: {
    ownerId: string;
    idempotencyKey: string;
    templateId: StudioAgentRecord["templateId"];
    configuration: StudioAgentConfiguration;
  }): StudioAgentRecord {
    return this.database.transaction(() => {
      const existing = this.database.raw.prepare(
        "SELECT * FROM studio_agents WHERE owner_id = ? AND create_idempotency_key = ?",
      ).get(input.ownerId, input.idempotencyKey);
      const configuration = studioAgentConfigurationSchema.parse(input.configuration);
      if (existing) {
        const agent = agentFromRow(existing);
        if (agent.templateId !== input.templateId || canonicalJson(agent.configuration) !== canonicalJson(configuration)) throw new StudioAgentError("studio_create_idempotency_conflict", "The creation key was already used for a different draft.", 409);
        return agent;
      }
      const now = Date.now();
      const id = `agt_${randomUUID().replaceAll("-", "")}`;
      this.database.raw.prepare(
        `INSERT INTO studio_agents(
           id, owner_id, create_idempotency_key, template_id, status,
           operational_state, draft_version, configuration_json,
           published_revision_id, created_at, updated_at, archived_at
         ) VALUES (?, ?, ?, ?, 'draft', 'draft', 1, ?, NULL, ?, ?, NULL)`,
      ).run(id, input.ownerId, input.idempotencyKey, input.templateId, canonicalJson(configuration), now, now);
      this.appendEvent(id, input.ownerId, "agent.created", { templateId: input.templateId });
      this.queueAgent(id);
      return this.requireOwned(id, input.ownerId);
    });
  }

  list(ownerId: string): StudioAgentRecord[] {
    return (this.database.raw.prepare(
      "SELECT * FROM studio_agents WHERE owner_id = ? ORDER BY updated_at DESC, id",
    ).all(ownerId) as unknown[]).map(agentFromRow);
  }

  get(agentId: string, ownerId: string): StudioAgentRecord | null {
    const row = this.database.raw.prepare(
      "SELECT * FROM studio_agents WHERE id = ? AND owner_id = ?",
    ).get(agentId, ownerId);
    return row ? agentFromRow(row) : null;
  }

  update(agentId: string, ownerId: string, expectedVersion: number, next: StudioAgentConfiguration): StudioAgentRecord {
    return this.database.transaction(() => {
      const current = this.requireOwned(agentId, ownerId);
      if (current.status === "archived") throw new StudioAgentError("studio_agent_archived", "Archived agents cannot be edited.", 409);
      if (current.draftVersion !== expectedVersion) throw new StudioAgentError("studio_agent_version_conflict", "The Studio draft changed in another session.", 409);
      const configuration = studioAgentConfigurationSchema.parse(next);
      const now = Date.now();
      const changed = this.database.raw.prepare(
        `UPDATE studio_agents SET configuration_json = ?, draft_version = draft_version + 1,
           status = 'draft', operational_state = 'draft', updated_at = ?
         WHERE id = ? AND owner_id = ? AND draft_version = ?`,
      ).run(canonicalJson(configuration), now, agentId, ownerId, expectedVersion);
      if (Number(changed.changes) !== 1) throw new StudioAgentError("studio_agent_version_conflict", "The Studio draft changed in another session.", 409);
      this.appendEvent(agentId, ownerId, "draft.updated", { fromVersion: expectedVersion, toVersion: expectedVersion + 1 });
      this.queueAgent(agentId);
      return this.requireOwned(agentId, ownerId);
    });
  }

  publish(input: {
    agentId: string;
    ownerId: string;
    expectedVersion: number;
    idempotencyKey: string;
    channels: StudioChannel[];
    hasCapacity: boolean;
  }): { agent: StudioAgentRecord; revision: StudioAgentRevision; deployments: StudioChannelDeployment[] } {
    return this.database.transaction(() => {
      const replayRows = this.database.raw.prepare(
        "SELECT * FROM studio_channel_deployments WHERE owner_id = ? AND publish_idempotency_key = ? ORDER BY channel",
      ).all(input.ownerId, input.idempotencyKey) as unknown[];
      if (replayRows.length > 0) {
        const deployments = replayRows.map(deploymentFromRow);
        const agent = this.requireOwned(deployments[0]!.agentId, input.ownerId);
        if (agent.id !== input.agentId || deployments.length !== input.channels.length || deployments.some((item) => !input.channels.includes(item.channel))) {
          throw new StudioAgentError("studio_publish_idempotency_conflict", "The publication key was already used for a different request.", 409);
        }
        return { agent, revision: this.requireRevision(deployments[0]!.revisionId, agent.id, input.ownerId), deployments };
      }
      const agent = this.requireOwned(input.agentId, input.ownerId);
      if (agent.status === "archived") throw new StudioAgentError("studio_agent_archived", "Archived agents cannot be published.", 409);
      if (agent.draftVersion !== input.expectedVersion) throw new StudioAgentError("studio_agent_version_conflict", "The Studio draft changed before publication.", 409);
      if (input.channels.length === 0 || new Set(input.channels).size !== input.channels.length) {
        throw new StudioAgentError("studio_publish_channels_invalid", "Choose one or more unique publication channels.");
      }
      const revision = this.createRevision(agent);
      const now = Date.now();
      for (const channel of input.channels) {
        this.database.raw.prepare(
          `UPDATE studio_channel_deployments SET state = 'revoked', revoked_at = ?, updated_at = ?
           WHERE agent_id = ? AND channel = ? AND revoked_at IS NULL`,
        ).run(now, now, agent.id, channel);
      }
      const state = input.hasCapacity ? "ready" : "waiting_for_capacity";
      const deployments = input.channels.map((channel) => {
        const id = `dep_${randomUUID().replaceAll("-", "")}`;
        const publicId = `agent_${randomUUID().replaceAll("-", "").slice(0, 24)}`;
        this.database.raw.prepare(
          `INSERT INTO studio_channel_deployments(
             id, agent_id, revision_id, owner_id, channel, state, public_id,
             publish_idempotency_key, created_at, updated_at, revoked_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)`,
        ).run(id, agent.id, revision.id, input.ownerId, channel, state, publicId, input.idempotencyKey, now, now);
        return deploymentFromRow(this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE id = ?").get(id)!);
      });
      this.database.raw.prepare(
        `UPDATE studio_agents SET status = 'published', operational_state = ?,
           published_revision_id = ?, updated_at = ? WHERE id = ? AND owner_id = ?`,
      ).run(state, revision.id, now, agent.id, input.ownerId);
      this.appendEvent(agent.id, input.ownerId, "agent.published", { revisionId: revision.id, channels: input.channels, state });
      this.queueAgent(agent.id);
      this.queueRevision(revision.id);
      for (const deployment of deployments) this.queueDeployment(deployment.id);
      return { agent: this.requireOwned(agent.id, input.ownerId), revision, deployments };
    });
  }

  archive(agentId: string, ownerId: string): StudioAgentRecord {
    return this.database.transaction(() => {
      this.requireOwned(agentId, ownerId);
      const now = Date.now();
      this.database.raw.prepare(
        "UPDATE studio_channel_deployments SET state = 'revoked', revoked_at = COALESCE(revoked_at, ?), updated_at = ? WHERE agent_id = ? AND owner_id = ? AND revoked_at IS NULL",
      ).run(now, now, agentId, ownerId);
      this.database.raw.prepare(
        "UPDATE studio_agents SET status = 'archived', operational_state = 'revoked', archived_at = ?, updated_at = ? WHERE id = ? AND owner_id = ?",
      ).run(now, now, agentId, ownerId);
      this.appendEvent(agentId, ownerId, "agent.archived", {});
      this.queueAgent(agentId);
      for (const deployment of this.deployments(agentId, ownerId)) this.queueDeployment(deployment.id);
      return this.requireOwned(agentId, ownerId);
    });
  }

  revisions(agentId: string, ownerId: string): StudioAgentRevision[] {
    this.requireOwned(agentId, ownerId);
    return (this.database.raw.prepare(
      "SELECT * FROM studio_agent_revisions WHERE agent_id = ? AND owner_id = ? ORDER BY revision DESC",
    ).all(agentId, ownerId) as unknown[]).map(revisionFromRow);
  }

  deployments(agentId: string, ownerId: string): StudioChannelDeployment[] {
    this.requireOwned(agentId, ownerId);
    return (this.database.raw.prepare(
      "SELECT * FROM studio_channel_deployments WHERE agent_id = ? AND owner_id = ? ORDER BY created_at DESC",
    ).all(agentId, ownerId) as unknown[]).map(deploymentFromRow);
  }

  revision(revisionId: string, agentId: string, ownerId: string): StudioAgentRevision {
    return this.requireRevision(revisionId, agentId, ownerId);
  }

  deploymentByPublicId(publicId: string, channel?: StudioChannel): StudioChannelDeployment | null {
    const row = channel
      ? this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE public_id = ? AND channel = ? AND revoked_at IS NULL").get(publicId, channel)
      : this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE public_id = ? AND revoked_at IS NULL").get(publicId);
    return row ? deploymentFromRow(row) : null;
  }

  revokeDeployment(deploymentId: string, ownerId: string): StudioChannelDeployment {
    return this.database.transaction(() => {
      const row = this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE id = ? AND owner_id = ?").get(deploymentId, ownerId);
      if (!row) throw new StudioAgentError("studio_deployment_not_found", "Studio deployment not found.", 404);
      const current = deploymentFromRow(row);
      if (!current.revokedAt) {
        const now = Date.now();
        this.database.raw.prepare("UPDATE studio_channel_deployments SET state = 'revoked', revoked_at = ?, updated_at = ? WHERE id = ? AND owner_id = ?").run(now, now, deploymentId, ownerId);
        this.appendEvent(current.agentId, ownerId, "deployment.revoked", { deploymentId });
        this.queueDeployment(deploymentId);
      }
      return deploymentFromRow(this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE id = ?").get(deploymentId)!);
    });
  }

  rollback(input: { agentId: string; ownerId: string; revisionId: string; idempotencyKey: string; channels: StudioChannel[]; hasCapacity: boolean }): { agent: StudioAgentRecord; revision: StudioAgentRevision; deployments: StudioChannelDeployment[] } {
    return this.database.transaction(() => {
      const agent = this.requireOwned(input.agentId, input.ownerId);
      if (agent.status === "archived") throw new StudioAgentError("studio_agent_archived", "Archived agents cannot be rolled back.", 409);
      const revision = this.requireRevision(input.revisionId, agent.id, input.ownerId);
      const replay = this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE owner_id = ? AND publish_idempotency_key = ? ORDER BY channel").all(input.ownerId, input.idempotencyKey) as unknown[];
      if (replay.length) {
        const deployments = replay.map(deploymentFromRow);
        if (deployments.some((item) => item.agentId !== agent.id || item.revisionId !== revision.id || !input.channels.includes(item.channel)) || deployments.length !== input.channels.length) throw new StudioAgentError("studio_publish_idempotency_conflict", "Rollback key was already used for another request.", 409);
        return { agent: this.requireOwned(agent.id, input.ownerId), revision, deployments };
      }
      if (!input.channels.length || new Set(input.channels).size !== input.channels.length) throw new StudioAgentError("studio_publish_channels_invalid", "Choose unique rollback channels.");
      const now = Date.now();
      const state = input.hasCapacity ? "ready" : "waiting_for_capacity";
      for (const channel of input.channels) this.database.raw.prepare("UPDATE studio_channel_deployments SET state = 'revoked', revoked_at = ?, updated_at = ? WHERE agent_id = ? AND channel = ? AND revoked_at IS NULL").run(now, now, agent.id, channel);
      const deployments = input.channels.map((channel) => {
        const id = `dep_${randomUUID().replaceAll("-", "")}`;
        this.database.raw.prepare("INSERT INTO studio_channel_deployments(id, agent_id, revision_id, owner_id, channel, state, public_id, publish_idempotency_key, created_at, updated_at, revoked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)").run(id, agent.id, revision.id, input.ownerId, channel, state, `agent_${randomUUID().replaceAll("-", "").slice(0, 24)}`, input.idempotencyKey, now, now);
        this.queueDeployment(id);
        return deploymentFromRow(this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE id = ?").get(id)!);
      });
      this.database.raw.prepare("UPDATE studio_agents SET status = 'published', operational_state = ?, published_revision_id = ?, updated_at = ? WHERE id = ? AND owner_id = ?").run(state, revision.id, now, agent.id, input.ownerId);
      this.appendEvent(agent.id, input.ownerId, "agent.rolled_back", { revisionId: revision.id, channels: input.channels, state });
      this.queueAgent(agent.id);
      return { agent: this.requireOwned(agent.id, input.ownerId), revision, deployments };
    });
  }

  activateCompatibleWaiting(modelId: string): number {
    return this.database.transaction(() => {
      const rows = this.database.raw.prepare(`SELECT d.id, d.agent_id, d.owner_id FROM studio_channel_deployments d
        JOIN studio_agent_revisions r ON r.id = d.revision_id
        WHERE d.state = 'waiting_for_capacity' AND d.revoked_at IS NULL
        AND json_extract(r.configuration_json, '$.modelPolicy.preferredModel') = ?`).all(modelId) as Array<{ id: string; agent_id: string; owner_id: string }>;
      const now = Date.now();
      for (const row of rows) {
        this.database.raw.prepare("UPDATE studio_channel_deployments SET state = 'ready', updated_at = ? WHERE id = ? AND state = 'waiting_for_capacity'").run(now, row.id);
        this.database.raw.prepare("UPDATE studio_agents SET operational_state = 'ready', updated_at = ? WHERE id = ? AND owner_id = ? AND operational_state = 'waiting_for_capacity'").run(now, row.agent_id, row.owner_id);
        this.appendEvent(row.agent_id, row.owner_id, "deployment.ready", { deploymentId: row.id, modelId });
        this.queueDeployment(row.id); this.queueAgent(row.agent_id);
      }
      return rows.length;
    });
  }

  private createRevision(agent: StudioAgentRecord): StudioAgentRevision {
    const configurationJson = canonicalJson(agent.configuration);
    const digest = createHash("sha256").update("mycellios-studio-revision/1\0").update(configurationJson).digest("hex");
    const existing = this.database.raw.prepare(
      "SELECT * FROM studio_agent_revisions WHERE agent_id = ? AND configuration_digest = ?",
    ).get(agent.id, digest);
    if (existing) return revisionFromRow(existing);
    const next = this.database.raw.prepare(
      "SELECT COALESCE(MAX(revision), 0) + 1 AS revision FROM studio_agent_revisions WHERE agent_id = ?",
    ).get(agent.id) as { revision: number };
    const id = `rev_${randomUUID().replaceAll("-", "")}`;
    const createdAt = Date.now();
    this.database.raw.prepare(
      `INSERT INTO studio_agent_revisions(id, agent_id, owner_id, revision, configuration_json,
         configuration_digest, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(id, agent.id, agent.ownerId, Number(next.revision), configurationJson, digest, createdAt);
    return this.requireRevision(id, agent.id, agent.ownerId);
  }

  private requireOwned(agentId: string, ownerId: string): StudioAgentRecord {
    const agent = this.get(agentId, ownerId);
    if (!agent) throw new StudioAgentError("studio_agent_not_found", "Studio agent not found.", 404);
    return agent;
  }

  private requireRevision(revisionId: string, agentId: string, ownerId: string): StudioAgentRevision {
    const row = this.database.raw.prepare(
      "SELECT * FROM studio_agent_revisions WHERE id = ? AND agent_id = ? AND owner_id = ?",
    ).get(revisionId, agentId, ownerId);
    if (!row) throw new StudioAgentError("studio_revision_not_found", "Studio revision not found.", 404);
    return revisionFromRow(row);
  }

  private appendEvent(agentId: string, ownerId: string, eventType: string, details: Record<string, unknown>): void {
    const previous = this.database.raw.prepare(
      "SELECT event_digest FROM studio_agent_events WHERE agent_id = ? ORDER BY sequence DESC LIMIT 1",
    ).get(agentId) as { event_digest: string } | undefined;
    const occurredAt = Date.now();
    const detailsJson = canonicalJson(details);
    const digest = createHash("sha256")
      .update("mycellios-studio-event/1\0")
      .update(previous?.event_digest ?? "genesis").update("\0")
      .update(agentId).update("\0").update(ownerId).update("\0")
      .update(eventType).update("\0").update(detailsJson).update("\0").update(String(occurredAt))
      .digest("hex");
    this.database.raw.prepare(
      `INSERT INTO studio_agent_events(agent_id, owner_id, event_type, details_json,
         previous_event_digest, event_digest, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?)`,
    ).run(agentId, ownerId, eventType, detailsJson, previous?.event_digest ?? null, digest, occurredAt);
    this.database.enqueueRemoteChange("studio_agent_events", digest, "upsert", {
      event_digest: digest, sequence: Number((this.database.raw.prepare("SELECT sequence FROM studio_agent_events WHERE event_digest = ?").get(digest) as { sequence: number }).sequence),
      agent_id: agentId, owner_id: ownerId, event_type: eventType, details,
      previous_event_digest: previous?.event_digest ?? null, occurred_at: new Date(occurredAt).toISOString(),
    });
  }

  private queueAgent(agentId: string): void {
    const row = this.database.raw.prepare("SELECT * FROM studio_agents WHERE id = ?").get(agentId) as Record<string, unknown> | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("studio_agents", agentId, "upsert", {
      id: row.id, owner_id: row.owner_id, create_idempotency_key: row.create_idempotency_key,
      template_id: row.template_id, status: row.status, operational_state: row.operational_state,
      draft_version: Number(row.draft_version), configuration: JSON.parse(String(row.configuration_json)),
      published_revision_id: row.published_revision_id,
      created_at: new Date(Number(row.created_at)).toISOString(), updated_at: new Date(Number(row.updated_at)).toISOString(),
      archived_at: row.archived_at === null ? null : new Date(Number(row.archived_at)).toISOString(),
    });
  }

  private queueRevision(revisionId: string): void {
    const row = this.database.raw.prepare("SELECT * FROM studio_agent_revisions WHERE id = ?").get(revisionId) as Record<string, unknown> | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("studio_agent_revisions", revisionId, "upsert", {
      id: row.id, agent_id: row.agent_id, owner_id: row.owner_id, revision: Number(row.revision),
      configuration: JSON.parse(String(row.configuration_json)), configuration_digest: row.configuration_digest,
      created_at: new Date(Number(row.created_at)).toISOString(),
    });
  }

  private queueDeployment(deploymentId: string): void {
    const row = this.database.raw.prepare("SELECT * FROM studio_channel_deployments WHERE id = ?").get(deploymentId) as Record<string, unknown> | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("studio_channel_deployments", deploymentId, "upsert", {
      id: row.id, agent_id: row.agent_id, revision_id: row.revision_id, owner_id: row.owner_id,
      channel: row.channel, state: row.state, public_id: row.public_id, publish_idempotency_key: row.publish_idempotency_key,
      created_at: new Date(Number(row.created_at)).toISOString(), updated_at: new Date(Number(row.updated_at)).toISOString(),
      revoked_at: row.revoked_at === null ? null : new Date(Number(row.revoked_at)).toISOString(),
    });
  }
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortValue(value));
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortValue);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, sortValue(item)]));
}

function agentFromRow(source: unknown): StudioAgentRecord {
  const row = source as Record<string, unknown>;
  return {
    id: String(row.id), ownerId: String(row.owner_id),
    templateId: row.template_id as StudioAgentRecord["templateId"],
    status: row.status as StudioAgentRecord["status"],
    operationalState: row.operational_state as StudioAgentRecord["operationalState"],
    draftVersion: Number(row.draft_version),
    configuration: studioAgentConfigurationSchema.parse(JSON.parse(String(row.configuration_json))),
    publishedRevisionId: row.published_revision_id === null ? null : String(row.published_revision_id),
    createdAt: Number(row.created_at), updatedAt: Number(row.updated_at),
    archivedAt: row.archived_at === null ? null : Number(row.archived_at),
  };
}

function revisionFromRow(source: unknown): StudioAgentRevision {
  const row = source as Record<string, unknown>;
  return {
    id: String(row.id), agentId: String(row.agent_id), ownerId: String(row.owner_id),
    revision: Number(row.revision), configuration: studioAgentConfigurationSchema.parse(JSON.parse(String(row.configuration_json))),
    digest: String(row.configuration_digest), createdAt: Number(row.created_at),
  };
}

function deploymentFromRow(source: unknown): StudioChannelDeployment {
  const row = source as Record<string, unknown>;
  return {
    id: String(row.id), agentId: String(row.agent_id), revisionId: String(row.revision_id), ownerId: String(row.owner_id),
    channel: row.channel as StudioChannelDeployment["channel"], state: row.state as StudioChannelDeployment["state"],
    publicId: String(row.public_id), createdAt: Number(row.created_at), updatedAt: Number(row.updated_at),
    revokedAt: row.revoked_at === null ? null : Number(row.revoked_at),
  };
}
