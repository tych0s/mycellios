import { randomBytes } from "node:crypto";
import { mkdir, open, readFile, rename, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { z } from "zod";
import { NODE_EVENT_ORIGIN_CURSOR } from "../contracts/node-control.js";

const schema = z.object({
  schema: z.literal("mycellios-node-reconciliation-state/1"),
  nodeId: z.string().min(1).max(128),
  generation: z.number().int().positive(),
  cursor: z.string().regex(/^evt_[0-9]{1,20}_[a-f0-9]{16}$/),
  updatedAt: z.string().datetime(),
}).strict();
export type NodeReconciliationState = z.infer<typeof schema>;

export class NodeReconciliationStateStore {
  readonly path: string;
  private chain: Promise<unknown> = Promise.resolve();
  constructor(path: string, private readonly nodeId: string) { this.path = resolve(path); }

  async load(generation: number): Promise<NodeReconciliationState> {
    try {
      const parsed = schema.parse(JSON.parse(await readFile(this.path, "utf8")));
      if (parsed.nodeId !== this.nodeId) throw new Error("node_reconciliation_state_wrong_node");
      return parsed.generation === generation ? parsed : this.initial(generation);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return this.initial(generation);
      if (error instanceof Error && error.message === "node_reconciliation_state_wrong_node") throw error;
      throw new Error("node_reconciliation_state_is_invalid", { cause: error });
    }
  }

  save(generation: number, cursor: string): Promise<NodeReconciliationState> {
    const write = this.chain.then(async () => {
      const next = schema.parse({ schema: "mycellios-node-reconciliation-state/1", nodeId: this.nodeId, generation, cursor, updatedAt: new Date().toISOString() });
      await mkdir(dirname(this.path), { recursive: true, mode: 0o700 });
      const temporary = `${this.path}.tmp-${process.pid}-${randomBytes(5).toString("hex")}`;
      try {
        const file = await open(temporary, "wx", 0o600);
        try { await file.writeFile(`${JSON.stringify(next, null, 2)}\n`); await file.sync(); } finally { await file.close(); }
        await rename(temporary, this.path);
      } finally { await rm(temporary, { force: true }); }
      return next;
    });
    this.chain = write.catch(() => undefined);
    return write;
  }

  private initial(generation: number): NodeReconciliationState {
    return schema.parse({ schema: "mycellios-node-reconciliation-state/1", nodeId: this.nodeId, generation, cursor: NODE_EVENT_ORIGIN_CURSOR, updatedAt: new Date(0).toISOString() });
  }
}
