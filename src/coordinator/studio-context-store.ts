import { createHash, randomUUID } from "node:crypto";
import type { MeshDatabase } from "../storage/database.js";
import { StudioAgentError } from "./studio-agent-store.js";

const MAX_DOCUMENT_BYTES = 1_000_000;
const MAX_CHUNK_CHARS = 1_200;
const MAX_QUERY_CHARS = 500;

export interface StudioKnowledgeMatch {
  sourceId: string;
  sourceName: string;
  chunkId: string;
  ordinal: number;
  content: string;
}

export interface StudioMemoryFact {
  id: string;
  subjectId: string;
  fact: string;
  status: "proposed" | "approved" | "rejected" | "expired" | "deleted";
  origin: string;
  confidence: number;
  expiresAt: number | null;
  createdAt: number;
  updatedAt: number;
}

export class StudioContextStore {
  constructor(private readonly database: MeshDatabase) {}

  ingestText(input: { ownerId: string; agentId: string; name: string; mediaType: string; content: string }): { id: string; chunks: number; digest: string } {
    this.requireAgent(input.ownerId, input.agentId);
    const bytes = Buffer.byteLength(input.content, "utf8");
    if (!input.name.trim() || input.name.length > 200) throw new StudioAgentError("studio_source_name_invalid", "Knowledge source name is invalid.");
    if (!new Set(["text/plain", "text/markdown"]).has(input.mediaType)) throw new StudioAgentError("studio_source_media_type_unsupported", "Only UTF-8 text and Markdown are accepted.", 415);
    if (bytes === 0 || bytes > MAX_DOCUMENT_BYTES || input.content.includes("\0")) throw new StudioAgentError("studio_source_content_invalid", "Knowledge source is empty, too large, or not valid text.", 413);
    const digest = sha(input.content);
    return this.database.transaction(() => {
      const existing = this.database.raw.prepare("SELECT id FROM studio_knowledge_sources WHERE agent_id = ? AND owner_id = ? AND content_sha256 = ? AND state != 'deleted'").get(input.agentId, input.ownerId, digest) as { id: string } | undefined;
      if (existing) {
        const count = this.database.raw.prepare("SELECT COUNT(*) AS count FROM studio_knowledge_chunks WHERE source_id = ?").get(existing.id) as { count: number };
        return { id: existing.id, chunks: Number(count.count), digest };
      }
      const id = `src_${randomUUID().replaceAll("-", "")}`;
      const now = Date.now();
      const chunks = splitDocument(input.content);
      this.database.raw.prepare("INSERT INTO studio_knowledge_sources(id, agent_id, owner_id, name, media_type, content_sha256, size_bytes, state, created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?, ?, NULL)").run(id, input.agentId, input.ownerId, input.name.trim(), input.mediaType, digest, bytes, now, now);
      chunks.forEach((content, ordinal) => {
        const chunkId = `chk_${randomUUID().replaceAll("-", "")}`;
        this.database.raw.prepare("INSERT INTO studio_knowledge_chunks(id, source_id, agent_id, owner_id, ordinal, content, content_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)").run(chunkId, id, input.agentId, input.ownerId, ordinal, content, sha(content), now);
        this.database.raw.prepare("INSERT INTO studio_knowledge_fts(chunk_id, owner_id, agent_id, source_id, content) VALUES (?, ?, ?, ?, ?)").run(chunkId, input.ownerId, input.agentId, id, content);
        this.database.enqueueRemoteChange("studio_knowledge_chunks", chunkId, "upsert", { id: chunkId, source_id: id, agent_id: input.agentId, owner_id: input.ownerId, ordinal, content, content_sha256: sha(content), created_at: new Date(now).toISOString() });
      });
      this.database.enqueueRemoteChange("studio_knowledge_sources", id, "upsert", { id, agent_id: input.agentId, owner_id: input.ownerId, name: input.name.trim(), media_type: input.mediaType, content_sha256: digest, size_bytes: bytes, state: "ready", created_at: new Date(now).toISOString(), updated_at: new Date(now).toISOString(), deleted_at: null });
      return { id, chunks: chunks.length, digest };
    });
  }

  retrieve(ownerId: string, agentId: string, query: string, limit = 5): StudioKnowledgeMatch[] {
    this.requireAgent(ownerId, agentId);
    const terms = tokenizeQuery(query);
    if (terms.length === 0) return [];
    const boundedLimit = Math.max(1, Math.min(10, Math.trunc(limit)));
    return (this.database.raw.prepare(`SELECT c.id, c.source_id, c.ordinal, c.content, s.name
      FROM studio_knowledge_fts f JOIN studio_knowledge_chunks c ON c.id = f.chunk_id
      JOIN studio_knowledge_sources s ON s.id = c.source_id
      WHERE f.owner_id = ? AND f.agent_id = ? AND s.state = 'ready' AND studio_knowledge_fts MATCH ?
      ORDER BY bm25(studio_knowledge_fts), c.ordinal LIMIT ?`).all(ownerId, agentId, terms.map((term) => `"${term}"`).join(" OR "), boundedLimit) as Array<Record<string, unknown>>).map((row) => ({
        sourceId: String(row.source_id), sourceName: String(row.name), chunkId: String(row.id), ordinal: Number(row.ordinal), content: String(row.content),
      }));
  }

  deleteSource(ownerId: string, agentId: string, sourceId: string): void {
    this.requireAgent(ownerId, agentId);
    this.database.transaction(() => {
      const owned = this.database.raw.prepare("SELECT id FROM studio_knowledge_sources WHERE id = ? AND agent_id = ? AND owner_id = ? AND state != 'deleted'").get(sourceId, agentId, ownerId);
      if (!owned) throw new StudioAgentError("studio_source_not_found", "Knowledge source not found.", 404);
      this.database.raw.prepare("DELETE FROM studio_knowledge_fts WHERE source_id = ? AND owner_id = ? AND agent_id = ?").run(sourceId, ownerId, agentId);
      this.database.raw.prepare("DELETE FROM studio_knowledge_chunks WHERE source_id = ? AND owner_id = ? AND agent_id = ?").run(sourceId, ownerId, agentId);
      this.database.raw.prepare("UPDATE studio_knowledge_sources SET state = 'deleted', deleted_at = ?, updated_at = ? WHERE id = ?").run(Date.now(), Date.now(), sourceId);
      this.database.enqueueRemoteChange("studio_knowledge_sources", sourceId, "upsert", { id: sourceId, agent_id: agentId, owner_id: ownerId, state: "deleted", deleted_at: new Date().toISOString(), updated_at: new Date().toISOString() });
    });
  }

  addFact(input: { ownerId: string; agentId: string; subjectId: string; fact: string; origin: string; confidence: number; approved: boolean; expiresAt?: number | null | undefined }): StudioMemoryFact {
    this.requireAgent(input.ownerId, input.agentId);
    if (!input.subjectId.trim() || input.subjectId.length > 128 || !input.fact.trim() || input.fact.length > 1_000 || !input.origin.trim() || input.origin.length > 200 || !Number.isFinite(input.confidence) || input.confidence < 0 || input.confidence > 1) throw new StudioAgentError("studio_memory_fact_invalid", "Memory fact is outside allowed bounds.");
    const id = `mem_${randomUUID().replaceAll("-", "")}`;
    const now = Date.now();
    this.database.raw.prepare("INSERT INTO studio_memory_facts(id, agent_id, owner_id, subject_id, fact, status, origin, confidence, expires_at, created_at, updated_at, deleted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)").run(id, input.agentId, input.ownerId, input.subjectId.trim(), input.fact.trim(), input.approved ? "approved" : "proposed", input.origin.trim(), input.confidence, input.expiresAt ?? null, now, now);
    this.queueFact(id);
    return this.getFact(input.ownerId, input.agentId, id);
  }

  approveFact(ownerId: string, agentId: string, factId: string): StudioMemoryFact {
    this.requireAgent(ownerId, agentId);
    const changed = this.database.raw.prepare("UPDATE studio_memory_facts SET status = 'approved', updated_at = ? WHERE id = ? AND agent_id = ? AND owner_id = ? AND status = 'proposed'").run(Date.now(), factId, agentId, ownerId);
    if (Number(changed.changes) !== 1) throw new StudioAgentError("studio_memory_fact_not_found", "Proposed memory fact not found.", 404);
    this.queueFact(factId);
    return this.getFact(ownerId, agentId, factId);
  }

  listFacts(ownerId: string, agentId: string, subjectId: string, approvedOnly = false): StudioMemoryFact[] {
    this.requireAgent(ownerId, agentId);
    const now = Date.now();
    this.database.raw.prepare("UPDATE studio_memory_facts SET status = 'expired', updated_at = ? WHERE owner_id = ? AND agent_id = ? AND status = 'approved' AND expires_at IS NOT NULL AND expires_at <= ?").run(now, ownerId, agentId, now);
    const rows = this.database.raw.prepare(`SELECT * FROM studio_memory_facts WHERE owner_id = ? AND agent_id = ? AND subject_id = ? AND status ${approvedOnly ? "= 'approved'" : "NOT IN ('deleted', 'expired')"} ORDER BY updated_at DESC, id`).all(ownerId, agentId, subjectId) as unknown[];
    return rows.map(factFromRow);
  }

  deleteFact(ownerId: string, agentId: string, factId: string): void {
    this.requireAgent(ownerId, agentId);
    const changed = this.database.raw.prepare("UPDATE studio_memory_facts SET fact = '[deleted]', status = 'deleted', deleted_at = ?, updated_at = ? WHERE id = ? AND agent_id = ? AND owner_id = ? AND status != 'deleted'").run(Date.now(), Date.now(), factId, agentId, ownerId);
    if (Number(changed.changes) !== 1) throw new StudioAgentError("studio_memory_fact_not_found", "Memory fact not found.", 404);
    this.queueFact(factId);
  }

  executeTool(input: { ownerId: string; agentId: string; enabledTools: string[]; toolId: string; args: unknown }): { output: unknown; auditId: string } {
    this.requireAgent(input.ownerId, input.agentId);
    const started = Date.now();
    const auditId = `tool_${randomUUID().replaceAll("-", "")}`;
    let outcome: "completed" | "rejected" | "failed" = "rejected";
    let output: unknown;
    let errorCode: string | null = null;
    try {
      if (!input.enabledTools.includes(input.toolId)) throw new StudioAgentError("studio_tool_not_granted", "Tool is not granted.", 403);
      if (input.toolId === "documents") {
        const args = input.args as { query?: unknown; limit?: unknown };
        if (!args || typeof args.query !== "string") throw new StudioAgentError("studio_tool_input_invalid", "Document query is required.");
        output = this.retrieve(input.ownerId, input.agentId, args.query, typeof args.limit === "number" ? args.limit : 5);
      } else if (input.toolId === "calculator") {
        const expression = (input.args as { expression?: unknown })?.expression;
        if (typeof expression !== "string" || expression.length > 200) throw new StudioAgentError("studio_tool_input_invalid", "A bounded expression is required.");
        output = { value: calculate(expression) };
      } else {
        throw new StudioAgentError("studio_tool_disabled", "Network tools are disabled until an explicit allowlist is configured.", 403);
      }
      outcome = "completed";
      return { output, auditId };
    } catch (error) {
      errorCode = error instanceof StudioAgentError ? error.code : "studio_tool_failed";
      outcome = error instanceof StudioAgentError ? "rejected" : "failed";
      throw error;
    } finally {
      this.database.raw.prepare("INSERT INTO studio_tool_audit(id, agent_id, owner_id, tool_id, input_digest, outcome, output_json, error_code, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)").run(auditId, input.agentId, input.ownerId, input.toolId, sha(JSON.stringify(input.args)), outcome, output === undefined ? null : JSON.stringify(output).slice(0, 16_000), errorCode, Math.max(0, Date.now() - started), Date.now());
      const row = this.database.raw.prepare("SELECT * FROM studio_tool_audit WHERE id = ?").get(auditId) as Record<string, unknown>;
      this.database.enqueueRemoteChange("studio_tool_audit", auditId, "upsert", { ...row, output: row.output_json === null ? null : JSON.parse(String(row.output_json)), output_json: undefined, created_at: new Date(Number(row.created_at)).toISOString() });
    }
  }

  private requireAgent(ownerId: string, agentId: string): void {
    if (!this.database.raw.prepare("SELECT 1 FROM studio_agents WHERE id = ? AND owner_id = ? AND status != 'archived'").get(agentId, ownerId)) throw new StudioAgentError("studio_agent_not_found", "Studio agent not found.", 404);
  }

  private getFact(ownerId: string, agentId: string, factId: string): StudioMemoryFact {
    const row = this.database.raw.prepare("SELECT * FROM studio_memory_facts WHERE id = ? AND agent_id = ? AND owner_id = ?").get(factId, agentId, ownerId);
    if (!row) throw new StudioAgentError("studio_memory_fact_not_found", "Memory fact not found.", 404);
    return factFromRow(row);
  }

  private queueFact(factId: string): void {
    const row = this.database.raw.prepare("SELECT * FROM studio_memory_facts WHERE id = ?").get(factId) as Record<string, unknown> | undefined;
    if (!row) return;
    this.database.enqueueRemoteChange("studio_memory_facts", factId, "upsert", { ...row, expires_at: row.expires_at === null ? null : new Date(Number(row.expires_at)).toISOString(), created_at: new Date(Number(row.created_at)).toISOString(), updated_at: new Date(Number(row.updated_at)).toISOString(), deleted_at: row.deleted_at === null ? null : new Date(Number(row.deleted_at)).toISOString() });
  }
}

function splitDocument(value: string): string[] {
  const normalized = value.replace(/\r\n?/g, "\n").replace(/[\t ]+/g, " ").trim();
  const chunks: string[] = [];
  for (let offset = 0; offset < normalized.length; offset += MAX_CHUNK_CHARS) chunks.push(normalized.slice(offset, offset + MAX_CHUNK_CHARS));
  return chunks;
}

function tokenizeQuery(value: string): string[] {
  return [...new Set(value.slice(0, MAX_QUERY_CHARS).normalize("NFKC").toLowerCase().match(/[\p{L}\p{N}]{2,}/gu) ?? [])].slice(0, 12);
}

function calculate(expression: string): number {
  const tokens = expression.replace(/\s+/g, "").match(/\d+(?:\.\d+)?|[()+\-*/]/g);
  if (!tokens || tokens.join("") !== expression.replace(/\s+/g, "")) throw new StudioAgentError("studio_calculator_expression_invalid", "Expression contains unsupported characters.");
  let index = 0;
  const primary = (): number => {
    const token = tokens[index++];
    if (token === "(") { const value = add(); if (tokens[index++] !== ")") throw new Error("parenthesis"); return value; }
    if (token === "-") return -primary();
    const value = Number(token);
    if (!Number.isFinite(value)) throw new Error("number");
    return value;
  };
  const multiply = (): number => { let value = primary(); while (tokens[index] === "*" || tokens[index] === "/") { const op = tokens[index++]; const right = primary(); value = op === "*" ? value * right : value / right; } return value; };
  const add = (): number => { let value = multiply(); while (tokens[index] === "+" || tokens[index] === "-") { const op = tokens[index++]; const right = multiply(); value = op === "+" ? value + right : value - right; } return value; };
  try { const value = add(); if (index !== tokens.length || !Number.isFinite(value)) throw new Error("result"); return value; } catch { throw new StudioAgentError("studio_calculator_expression_invalid", "Expression is invalid or non-finite."); }
}

function sha(value: string): string { return createHash("sha256").update(value).digest("hex"); }
function factFromRow(source: unknown): StudioMemoryFact {
  const row = source as Record<string, unknown>;
  return { id: String(row.id), subjectId: String(row.subject_id), fact: String(row.fact), status: row.status as StudioMemoryFact["status"], origin: String(row.origin), confidence: Number(row.confidence), expiresAt: row.expires_at === null ? null : Number(row.expires_at), createdAt: Number(row.created_at), updatedAt: Number(row.updated_at) };
}
