import { createHash, randomBytes, randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { z } from "zod";
import {
  nodeCommandResultSchema,
  nodeCommandSchema,
  parseAuthorizedNodeCommand,
  type NodeCommand,
  type NodeCommandResult,
} from "../contracts/node-control.js";

const journalRecordSchema = z.object({
  commandId: z.string().uuid(),
  nonce: z.string().min(1),
  commandDigest: z.string().regex(/^sha256:[a-f0-9]{64}$/),
  state: z.enum(["applying", "complete"]),
  operationKey: z.string().regex(/^node-command:[0-9a-f-]{36}$/),
  result: nodeCommandResultSchema.nullable(),
  updatedAt: z.string().datetime(),
}).strict();

const journalSchema = z.object({
  schema: z.literal("mycellios-node-command-journal/1"),
  nodeId: z.string().min(1).max(128),
  generation: z.number().int().positive(),
  records: z.array(journalRecordSchema).max(4_096),
}).strict();

type Journal = z.infer<typeof journalSchema>;

export interface NodeCommandApplication {
  /**
   * Stable across crash/replay. Implementations must use it as their
   * idempotency key whenever the underlying mutation is not declarative.
   */
  operationKey: string;
  command: NodeCommand;
}

export type NodeCommandApply = (application: NodeCommandApplication) => Promise<unknown>;

export class NodeCommandExecutionError extends Error {
  constructor(readonly code: string, message: string) {
    super(message);
    this.name = "NodeCommandExecutionError";
  }
}

/** Testable representation of a process dying after the mutation began. */
export class NodeCommandExecutionInterruptedError extends Error {
  constructor() {
    super("node_command_execution_interrupted");
    this.name = "NodeCommandExecutionInterruptedError";
  }
}

export class NodeCommandExecutor {
  readonly path: string;
  private chain: Promise<unknown> = Promise.resolve();

  constructor(
    path: string,
    private readonly nodeId: string,
    private readonly generation: number,
    private readonly now: () => number = Date.now,
  ) {
    if (!Number.isSafeInteger(generation) || generation < 1) throw new Error("node_command_generation_invalid");
    this.path = resolve(path);
  }

  execute(input: unknown, apply: NodeCommandApply): Promise<NodeCommandResult> {
    const run = this.chain.then(() => this.executeOnce(input, apply));
    this.chain = run.catch(() => undefined);
    return run;
  }

  async inspect(): Promise<Journal> {
    return this.load();
  }

  private async executeOnce(input: unknown, apply: NodeCommandApply): Promise<NodeCommandResult> {
    const parsed = nodeCommandSchema.parse(input);
    const commandJson = canonicalJson(parsed);
    const commandDigest = digest(commandJson);
    let journal = await this.load();
    const existing = journal.records.find(({ commandId }) => commandId === parsed.id);
    if (existing) {
      if (existing.commandDigest !== commandDigest) throw new Error("node_command_id_conflict");
      if (existing.state === "complete" && existing.result) return existing.result;
    }

    const consumedNonces = new Set(
      journal.records.filter(({ commandId }) => commandId !== parsed.id).map(({ nonce }) => nonce),
    );
    let command: NodeCommand;
    try {
      command = parseAuthorizedNodeCommand(parsed, {
        now: new Date(this.now()),
        minimumGeneration: this.generation,
        consumedNonces,
        expectedNodeId: this.nodeId,
      });
    } catch (error) {
      if (!(error instanceof Error) || error.message !== "node_command_expired") throw error;
      return this.finishWithoutApply(journal, parsed, commandDigest, "expired", {
        code: "node_command_expired",
        message: "The command expired before the node could apply it.",
      });
    }

    const operationKey = existing?.operationKey ?? `node-command:${command.id}`;
    if (!existing) {
      journal = appendBounded(journal, {
        commandId: command.id,
        nonce: command.nonce,
        commandDigest,
        state: "applying",
        operationKey,
        result: null,
        updatedAt: new Date(this.now()).toISOString(),
      });
      await this.write(journal);
    }

    try {
      const output = await apply({ command, operationKey });
      return await this.finish(journal, command.id, "applied", digest(canonicalJson(output ?? null)), null);
    } catch (error) {
      if (error instanceof NodeCommandExecutionInterruptedError) throw error;
      const execution = error instanceof NodeCommandExecutionError
        ? error
        : new NodeCommandExecutionError("node_command_apply_failed", safeErrorMessage(error));
      return this.finish(
        journal,
        command.id,
        "rejected",
        digest(canonicalJson({ code: execution.code })),
        { code: safeErrorCode(execution.code), message: safeErrorMessage(execution) },
      );
    }
  }

  private async finishWithoutApply(
    journal: Journal,
    command: NodeCommand,
    commandDigest: `sha256:${string}`,
    state: "expired",
    error: { code: string; message: string },
  ): Promise<NodeCommandResult> {
    const operationKey = `node-command:${command.id}`;
    if (!journal.records.some(({ commandId }) => commandId === command.id)) {
      journal = appendBounded(journal, {
        commandId: command.id,
        nonce: command.nonce,
        commandDigest,
        state: "applying",
        operationKey,
        result: null,
        updatedAt: new Date(this.now()).toISOString(),
      });
      await this.write(journal);
    }
    return this.finish(journal, command.id, state, digest(canonicalJson(error)), error);
  }

  private async finish(
    journal: Journal,
    commandId: string,
    state: "applied" | "rejected" | "expired",
    resultDigest: `sha256:${string}`,
    error: { code: string; message: string } | null,
  ): Promise<NodeCommandResult> {
    const result = nodeCommandResultSchema.parse({
      schema: "mycellios-node-command-result/1",
      id: randomUUID(),
      commandId,
      nodeId: this.nodeId,
      generation: this.generation,
      state,
      observedAt: new Date(this.now()).toISOString(),
      resultDigest,
      error,
    });
    const records = journal.records.map((record) => record.commandId === commandId ? {
      ...record,
      state: "complete" as const,
      result,
      updatedAt: new Date(this.now()).toISOString(),
    } : record);
    await this.write(journalSchema.parse({ ...journal, records }));
    return result;
  }

  private async load(): Promise<Journal> {
    try {
      const journal = journalSchema.parse(JSON.parse(await readFile(this.path, "utf8")));
      if (journal.nodeId !== this.nodeId) throw new Error("mycellios_node_command_journal_wrong_node");
      if (journal.generation !== this.generation) throw new Error("mycellios_node_command_journal_wrong_generation");
      return journal;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        if (error instanceof Error && error.message.startsWith("mycellios_node_command_journal_")) throw error;
        throw new Error("mycellios_node_command_journal_is_invalid", { cause: error });
      }
      return journalSchema.parse({
        schema: "mycellios-node-command-journal/1",
        nodeId: this.nodeId,
        generation: this.generation,
        records: [],
      });
    }
  }

  private async write(journal: Journal): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
    const temporary = `${this.path}.tmp-${process.pid}-${randomBytes(5).toString("hex")}`;
    try {
      const file = await open(temporary, "wx", 0o600);
      try {
        await file.writeFile(`${JSON.stringify(journal, null, 2)}\n`, "utf8");
        await file.sync();
      } finally {
        await file.close();
      }
      await rename(temporary, this.path);
    } finally {
      await rm(temporary, { force: true });
    }
  }
}

function appendBounded(journal: Journal, record: Journal["records"][number]): Journal {
  const complete = journal.records.filter(({ state }) => state === "complete");
  const applying = journal.records.filter(({ state }) => state === "applying");
  const retainedComplete = complete.slice(Math.max(0, complete.length - (4_096 - applying.length - 1)));
  return journalSchema.parse({ ...journal, records: [...retainedComplete, ...applying, record] });
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function digest(value: string): `sha256:${string}` {
  return `sha256:${createHash("sha256").update(value, "utf8").digest("hex")}`;
}

function safeErrorCode(value: string): string {
  return /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value) ? value : "node_command_apply_failed";
}

function safeErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (!message || message.length > 512 || /(token|secret|password|credential|authorization)/i.test(message)) {
    return "The node could not apply the command.";
  }
  return message;
}
