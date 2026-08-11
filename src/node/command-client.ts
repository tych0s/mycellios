import { z } from "zod";
import { nodeCommandSchema, nodeCommandResultSchema, nodeDesiredStateSchema, nodeEventSchema, nodeSnapshotSchema, type NodeCommand, type NodeSnapshot } from "../contracts/node-control.js";
import type { NodeCommandApply, NodeCommandExecutor } from "./command-executor.js";
import type { NodeReconciliationStateStore } from "./reconciliation-state.js";

const pulledCommandsSchema = z.object({
  object: z.literal("list"),
  data: z.array(z.object({
    command: nodeCommandSchema,
    state: z.enum(["queued", "delivered"]),
  }).strict()).max(256),
}).strict();
const reconciliationSchema = z.object({
  snapshot: nodeSnapshotSchema,
  desiredState: nodeDesiredStateSchema,
  events: z.array(nodeEventSchema),
  latestCursor: z.string().regex(/^evt_[0-9]{1,20}_[a-f0-9]{16}$/),
  acknowledgedCommandIds: z.array(z.string().uuid()).max(256).default([]),
}).strict();

export interface NodeCommandClientSession {
  token: string;
  generation: number;
}

export interface NodeCommandClientOptions {
  coordinatorUrl: string;
  nodeId: string;
  session: () => NodeCommandClientSession | null;
  executor: NodeCommandExecutor;
  apply: NodeCommandApply;
  signal: AbortSignal;
  fetch?: typeof fetch;
  pollIntervalMs?: number;
  retryIntervalMs?: number;
  onError?: (error: unknown) => void;
  onAcknowledged?: (command: NodeCommand, result: z.infer<typeof nodeCommandResultSchema>) => Promise<void> | void;
  reconciliation?: {
    state: NodeReconciliationStateStore;
    snapshot: (cursor: string, session: NodeCommandClientSession) => Promise<NodeSnapshot> | NodeSnapshot;
    onReconciled?: (result: z.infer<typeof reconciliationSchema>) => Promise<void> | void;
    onSnapshotAcknowledged?: (result: z.infer<typeof reconciliationSchema>) => Promise<void> | void;
  };
}

export async function runNodeCommandClient(options: NodeCommandClientOptions): Promise<void> {
  const request = options.fetch ?? fetch;
  const pollIntervalMs = Math.max(100, options.pollIntervalMs ?? 2_000);
  const retryIntervalMs = Math.max(100, options.retryIntervalMs ?? 1_000);
  while (!options.signal.aborted) {
    try {
      const session = options.session();
      if (!session) {
        await wait(pollIntervalMs, options.signal);
        continue;
      }
      const commands = await pullCommands(request, options, session);
      for (const { command } of commands) {
        if (options.signal.aborted) return;
        const result = await options.executor.execute(command, options.apply);
        await postResult(request, options, session, result);
        await options.onAcknowledged?.(command, result);
      }
      if (options.reconciliation) await postSnapshot(request, options, session);
      await wait(pollIntervalMs, options.signal);
    } catch (error) {
      if (options.signal.aborted) return;
      options.onError?.(error);
      await wait(retryIntervalMs, options.signal);
    }
  }
}

async function postSnapshot(request: typeof fetch, options: NodeCommandClientOptions, session: NodeCommandClientSession): Promise<void> {
  const reconciliation = options.reconciliation!;
  const state = await reconciliation.state.load(session.generation);
  const snapshot = nodeSnapshotSchema.parse(await reconciliation.snapshot(state.cursor, session));
  if (snapshot.nodeId !== options.nodeId || snapshot.generation !== session.generation || snapshot.cursor !== state.cursor) {
    throw new Error("node_snapshot_client_identity_mismatch");
  }
  const response = await request(endpoint(options.coordinatorUrl, options.nodeId, "snapshot"), {
    method: "POST",
    headers: { authorization: `Bearer ${session.token}`, "content-type": "application/json" },
    body: JSON.stringify(snapshot),
    signal: AbortSignal.any([options.signal, AbortSignal.timeout(10_000)]),
    redirect: "manual",
  });
  if (!response.ok) throw new Error(`node_snapshot_http_${response.status}`);
  const result = reconciliationSchema.parse(await responseJson(response));
  if (
    result.snapshot.nodeId !== snapshot.nodeId
    || result.snapshot.generation !== snapshot.generation
    || result.snapshot.cursor !== snapshot.cursor
    || result.desiredState.nodeId !== snapshot.nodeId
    || result.desiredState.generation !== snapshot.generation
  ) throw new Error("node_reconciliation_response_identity_mismatch");
  await reconciliation.onReconciled?.(result);
  await reconciliation.state.save(session.generation, result.latestCursor);
  await reconciliation.onSnapshotAcknowledged?.(result);
}

async function pullCommands(
  request: typeof fetch,
  options: NodeCommandClientOptions,
  session: NodeCommandClientSession,
): Promise<Array<{ command: NodeCommand; state: "queued" | "delivered" }>> {
  const url = endpoint(options.coordinatorUrl, options.nodeId, "commands");
  url.searchParams.set("generation", String(session.generation));
  url.searchParams.set("limit", "32");
  const response = await request(url, {
    method: "GET",
    headers: { authorization: `Bearer ${session.token}`, accept: "application/json" },
    signal: AbortSignal.any([options.signal, AbortSignal.timeout(10_000)]),
    redirect: "manual",
  });
  if (!response.ok) throw new Error(`node_command_pull_http_${response.status}`);
  return pulledCommandsSchema.parse(await responseJson(response)).data;
}

async function postResult(
  request: typeof fetch,
  options: NodeCommandClientOptions,
  session: NodeCommandClientSession,
  result: z.infer<typeof nodeCommandResultSchema>,
): Promise<void> {
  const response = await request(endpoint(options.coordinatorUrl, options.nodeId, "commands/results"), {
    method: "POST",
    headers: { authorization: `Bearer ${session.token}`, "content-type": "application/json" },
    body: JSON.stringify(result),
    signal: AbortSignal.any([options.signal, AbortSignal.timeout(10_000)]),
    redirect: "manual",
  });
  if (!response.ok) throw new Error(`node_command_result_http_${response.status}`);
  nodeCommandResultSchema.parse(await responseJson(response));
}

function endpoint(coordinatorUrl: string, nodeId: string, suffix: string): URL {
  const url = new URL(coordinatorUrl);
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("node_command_coordinator_protocol_invalid");
  url.pathname = `/internal/v1/nodes/${encodeURIComponent(nodeId)}/${suffix}`;
  url.search = "";
  url.hash = "";
  return url;
}

async function responseJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (Buffer.byteLength(text, "utf8") > 256 * 1024) throw new Error("node_command_response_too_large");
  try { return JSON.parse(text) as unknown; } catch { throw new Error("node_command_response_invalid_json"); }
}

function wait(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(finish, milliseconds);
    function finish() {
      clearTimeout(timer);
      signal.removeEventListener("abort", finish);
      resolve();
    }
    signal.addEventListener("abort", finish, { once: true });
  });
}
