import { randomBytes } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { z } from "zod";

const stateSchema = z.object({
  schema: z.literal("mycellios-node-control-state/2"),
  revision: z.number().int().nonnegative(),
  contributionEnabled: z.boolean(),
  draining: z.boolean(),
  updatedAt: z.string().datetime(),
}).strict();

const legacyStateSchema = z.object({
  schema: z.literal("mycellios-node-control-state/1"),
  revision: z.number().int().nonnegative(),
  contributionEnabled: z.boolean(),
  updatedAt: z.string().datetime(),
}).strict();

export type NodeControlState = z.infer<typeof stateSchema>;

export class NodeControlStateStore {
  readonly path: string;
  private writeChain: Promise<unknown> = Promise.resolve();

  constructor(path: string) { this.path = resolve(path); }

  async load(): Promise<NodeControlState> {
    try {
      const document = JSON.parse(await readFile(this.path, "utf8"));
      const current = stateSchema.safeParse(document);
      if (current.success) return current.data;
      const legacy = legacyStateSchema.safeParse(document);
      if (legacy.success) return stateSchema.parse({ ...legacy.data, schema: "mycellios-node-control-state/2", draining: false });
      throw current.error;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
        throw new Error("mycellios_node_control_state_is_invalid", { cause: error });
      }
      return {
        schema: "mycellios-node-control-state/2",
        revision: 0,
        contributionEnabled: true,
        draining: false,
        updatedAt: new Date(0).toISOString(),
      };
    }
  }

  setContributionEnabled(enabled: boolean): Promise<NodeControlState> {
    return this.mutate((current) => current.contributionEnabled === enabled ? current : {
      ...current,
      contributionEnabled: enabled,
    });
  }

  setDraining(draining: boolean): Promise<NodeControlState> {
    return this.mutate((current) => current.draining === draining ? current : {
      ...current,
      draining,
    });
  }

  private mutate(change: (current: NodeControlState) => NodeControlState): Promise<NodeControlState> {
    const write = this.writeChain.then(async () => {
      const current = await this.load();
      const changed = change(current);
      if (changed === current) return current;
      const next = stateSchema.parse({
        ...changed,
        revision: current.revision + 1,
        updatedAt: new Date().toISOString(),
      });
      await this.write(next);
      return next;
    });
    this.writeChain = write.catch(() => undefined);
    return write;
  }

  private async write(value: NodeControlState): Promise<void> {
    await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
    const temporary = `${this.path}.tmp-${process.pid}-${randomBytes(5).toString("hex")}`;
    try {
      const file = await open(temporary, "wx", 0o600);
      try {
        await file.writeFile(`${JSON.stringify(value, null, 2)}\n`, "utf8");
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
