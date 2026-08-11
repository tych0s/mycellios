import { randomBytes } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import { z } from "zod";

const lockSchema = z.object({
  schema: z.literal("mycellios-node-lock/1"),
  pid: z.number().int().positive(),
  token: z.string().regex(/^[0-9a-f]{32}$/),
  startedAt: z.string().datetime(),
}).strict();

export type NodeServiceState = "starting" | "ready" | "stopping" | "stopped" | "failed";

export class MycelliosNodeServiceHost {
  readonly stateDirectory: string;
  readonly lockPath: string;
  readonly healthPath: string;
  private token: string | null = null;

  constructor(stateDirectory: string) {
    this.stateDirectory = resolve(stateDirectory);
    this.lockPath = join(this.stateDirectory, "instance.lock");
    this.healthPath = join(this.stateDirectory, "health.json");
  }

  async acquire(configRevision: number): Promise<void> {
    if (this.token) return;
    await mkdir(this.stateDirectory, { recursive: true, mode: 0o700 });
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const token = randomBytes(16).toString("hex");
      try {
        const file = await open(this.lockPath, "wx", 0o600);
        try {
          await file.writeFile(`${JSON.stringify({
            schema: "mycellios-node-lock/1",
            pid: process.pid,
            token,
            startedAt: new Date().toISOString(),
          })}\n`, "utf8");
          await file.sync();
        } finally {
          await file.close();
        }
        this.token = token;
        await this.writeHealth("starting", configRevision);
        return;
      } catch (error) {
        if (!isAlreadyExists(error)) throw error;
        const existing = await readLock(this.lockPath);
        if (!existing) throw new Error("mycellios_node_instance_lock_is_invalid");
        if (existing && processExists(existing.pid)) {
          throw new Error(`mycellios_node_instance_is_already_running:${existing.pid}`);
        }
        await rm(this.lockPath, { force: true });
      }
    }
    throw new Error("mycellios_node_instance_lock_race");
  }

  async writeHealth(state: NodeServiceState, configRevision: number, error?: string): Promise<void> {
    if (!this.token) throw new Error("mycellios_node_service_lock_is_not_held");
    const temporary = `${this.healthPath}.tmp-${this.token}-${randomBytes(4).toString("hex")}`;
    const file = await open(temporary, "w", 0o600);
    try {
      await file.writeFile(`${JSON.stringify({
        schema: "mycellios-node-health/1",
        state,
        pid: process.pid,
        configRevision,
        updatedAt: new Date().toISOString(),
        ...(error ? { error: redactError(error) } : {}),
      }, null, 2)}\n`, "utf8");
      await file.sync();
    } finally {
      await file.close();
    }
    await rename(temporary, this.healthPath);
  }

  async release(configRevision: number, state: "stopped" | "failed" = "stopped", error?: string): Promise<void> {
    const token = this.token;
    if (!token) return;
    await this.writeHealth(state, configRevision, error);
    const current = await readLock(this.lockPath);
    if (current?.token === token) await rm(this.lockPath, { force: true });
    this.token = null;
  }
}

async function readLock(path: string): Promise<z.infer<typeof lockSchema> | null> {
  try { return lockSchema.parse(JSON.parse(await readFile(path, "utf8"))); } catch { return null; }
}

function processExists(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch (error) {
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

function isAlreadyExists(error: unknown): boolean {
  return (error as NodeJS.ErrnoException).code === "EEXIST";
}

function redactError(value: string): string {
  const candidate = value.match(/^[a-zA-Z0-9_.:-]{1,120}/)?.[0];
  if (
    !candidate
    || /(authorization|cookie|credential|password|secret|token|api[-_]?key)/i.test(candidate)
    || /[A-Za-z0-9_-]{32,}/.test(candidate)
  ) return "unclassified_error";
  return candidate;
}
