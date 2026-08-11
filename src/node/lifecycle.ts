import { randomUUID } from "node:crypto";
import { mkdir, open, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  NODE_LIFECYCLE_STATE_SCHEMA,
  nodeLifecycleStateSchema,
  type NodeLifecycleState,
} from "../contracts/node-lifecycle.js";

interface InstanceLockDocument {
  instanceId: string;
  pid: number;
  acquiredAt: string;
}

export interface NodeRuntimeHooks {
  start(): Promise<void>;
  drain(): Promise<void>;
  stop(): Promise<void>;
}

export interface NodeStartResult {
  started: boolean;
  state: NodeLifecycleState;
}

export class NodeLifecycleController {
  private ownedInstanceId: string | null = null;

  constructor(
    private readonly stateFile: string,
    private readonly lockFile: string,
    private readonly hooks: NodeRuntimeHooks,
    private readonly now: () => Date = () => new Date(),
    private readonly pid: number = process.pid,
    private readonly isProcessAlive: (pid: number) => boolean = processIsAlive,
  ) {}

  async readState(): Promise<NodeLifecycleState> {
    try {
      return nodeLifecycleStateSchema.parse(
        JSON.parse(await readFile(this.stateFile, "utf8")) as unknown,
      );
    } catch (error) {
      if (isMissing(error)) return initialState(this.now());
      throw new Error("node_lifecycle_state_load_failed", { cause: error });
    }
  }

  async ensureStarted(): Promise<NodeStartResult> {
    if (this.ownedInstanceId) {
      return { started: false, state: await this.readState() };
    }
    const lock = await this.acquireLock();
    if (!lock.owned) {
      return { started: false, state: await this.readState() };
    }
    this.ownedInstanceId = lock.document.instanceId;
    const previous = await this.readState();
    const startedAt = this.now().toISOString();
    const starting = await this.writeState({
      ...previous,
      generation: previous.generation + 1,
      status: "starting",
      instanceId: lock.document.instanceId,
      updatedAt: startedAt,
      lastStartedAt: startedAt,
      failureCode: null,
    });
    try {
      await this.hooks.start();
      return {
        started: true,
        state: await this.writeState({
          ...starting,
          status: "running",
          updatedAt: this.now().toISOString(),
        }),
      };
    } catch {
      const failed = await this.writeState({
        ...starting,
        status: "failed",
        updatedAt: this.now().toISOString(),
        failureCode: "runtime_start_failed",
      });
      await this.hooks.stop().catch(() => undefined);
      await this.releaseLock();
      throw new Error("node_runtime_start_failed", { cause: failed });
    }
  }

  async drainAndStop(): Promise<NodeLifecycleState> {
    if (!this.ownedInstanceId) throw new Error("node_lifecycle_not_lock_owner");
    const current = await this.readState();
    const draining = await this.writeState({
      ...current,
      status: "draining",
      updatedAt: this.now().toISOString(),
    });
    try {
      await this.hooks.drain();
      await this.hooks.stop();
      const stoppedAt = this.now().toISOString();
      const stopped = await this.writeState({
        ...draining,
        status: "stopped",
        instanceId: null,
        updatedAt: stoppedAt,
        lastStoppedAt: stoppedAt,
      });
      await this.releaseLock();
      return stopped;
    } catch {
      const failed = await this.writeState({
        ...draining,
        status: "failed",
        updatedAt: this.now().toISOString(),
        failureCode: "runtime_stop_failed",
      });
      throw new Error("node_runtime_stop_failed", { cause: failed });
    }
  }

  private async acquireLock(): Promise<{
    owned: boolean;
    document: InstanceLockDocument;
  }> {
    await mkdir(path.dirname(this.lockFile), { recursive: true });
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const document = {
        instanceId: randomUUID(),
        pid: this.pid,
        acquiredAt: this.now().toISOString(),
      };
      try {
        const handle = await open(this.lockFile, "wx", 0o600);
        await handle.writeFile(`${JSON.stringify(document)}\n`, "utf8");
        await handle.close();
        return { owned: true, document };
      } catch (error) {
        if (!isAlreadyExists(error)) throw new Error("node_lifecycle_lock_failed", { cause: error });
        const existing = await this.readLock();
        if (this.isProcessAlive(existing.pid)) return { owned: false, document: existing };
        await rm(this.lockFile, { force: true });
      }
    }
    throw new Error("node_lifecycle_lock_contended");
  }

  private async readLock(): Promise<InstanceLockDocument> {
    try {
      const value = JSON.parse(await readFile(this.lockFile, "utf8")) as Partial<InstanceLockDocument>;
      if (
        typeof value.instanceId !== "string" ||
        typeof value.pid !== "number" ||
        !Number.isInteger(value.pid) ||
        value.pid <= 0 ||
        typeof value.acquiredAt !== "string"
      ) throw new Error("invalid_lock");
      return value as InstanceLockDocument;
    } catch (error) {
      throw new Error("node_lifecycle_lock_corrupt", { cause: error });
    }
  }

  private async releaseLock(): Promise<void> {
    await rm(this.lockFile, { force: true });
    this.ownedInstanceId = null;
  }

  private async writeState(state: NodeLifecycleState): Promise<NodeLifecycleState> {
    const parsed = nodeLifecycleStateSchema.parse(state);
    await mkdir(path.dirname(this.stateFile), { recursive: true });
    const temporary = `${this.stateFile}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporary, `${JSON.stringify(parsed, null, 2)}\n`, {
        encoding: "utf8",
        flag: "wx",
        mode: 0o600,
      });
      await rename(temporary, this.stateFile);
      return parsed;
    } catch (error) {
      await rm(temporary, { force: true });
      throw error;
    }
  }
}

function initialState(now: Date): NodeLifecycleState {
  return {
    schema: NODE_LIFECYCLE_STATE_SCHEMA,
    version: 1,
    generation: 0,
    status: "stopped",
    instanceId: null,
    updatedAt: now.toISOString(),
    lastStartedAt: null,
    lastStoppedAt: null,
    failureCode: null,
  };
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return Boolean(
      error && typeof error === "object" && "code" in error && error.code === "EPERM"
    );
  }
}

function isMissing(error: unknown): boolean {
  return hasCode(error, "ENOENT");
}

function isAlreadyExists(error: unknown): boolean {
  return hasCode(error, "EEXIST");
}

function hasCode(error: unknown, code: string): boolean {
  return Boolean(error && typeof error === "object" && "code" in error && error.code === code);
}
