export interface ApiAccount {
  user_id: string;
  token_balance: number;
  usd_balance: string;
  lifetime_input_tokens: number;
  lifetime_output_tokens: number;
  request_count: number;
  limits: {
    requests_per_minute: number;
    max_concurrent: number;
    max_active_keys: number;
  };
  created_at: string;
  updated_at: string;
}

export interface ApiUsage {
  id: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  status: "pending" | "completed" | "failed";
  created_at: string;
  completed_at: string | null;
}

export interface ApiKeySummary {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

export interface CreatedApiKey extends ApiKeySummary {
  secret: string;
}

export interface OwnedNode {
  nodeId: string;
  status: "active" | "revoked";
  generation: number;
  credentialFingerprint: string;
  connected: boolean;
  workerId: string | null;
  observed: NodeSnapshot | null;
  desired: NodeDesiredState | null;
  commands: Array<{
    id: string;
    type: NodeCommand["type"];
    state: "queued" | "delivered" | "applied" | "rejected" | "expired";
    createdAt: string;
    completedAt: string | null;
  }>;
}

export interface OperatorEvidenceSnapshot {
  capturedAt: string;
  incidents: Array<{ modelId: string; incident: { code: string; title: string; summary: string; repairState: string; nextRetryAt: string | null } }>;
  releases: { configured: boolean; items: Array<{ channel: string; platform: string; arch: string; manifestId: string; sequence: number }> };
  certifications: { configured: boolean; items: Array<{ certificationId: string; modelFamily: string; decision: "certified" | "revoked"; topology: string; platform: string; backend: string; evidenceReceiptId: string; reviewedAt: string; expiresAt: string }>; blocker: string | null };
  receipts: Array<{ receiptId: string; jobId: string; routeClass: string; recoveryMode: string; traceDigest: string; completedAt: number }>;
}

export const configuredApiUrl = (apiBaseUrl?: string): string => apiBaseUrl ?? "/v1";

export async function loadApiAccount(accessToken: string, apiBaseUrl?: string): Promise<ApiAccount> {
  return apiRequest<ApiAccount>("/v1/account", accessToken, {}, apiBaseUrl);
}

export function apiRequestUrl(path: string, apiBaseUrl?: string): string {
  if (!apiBaseUrl) return path;
  const base = apiBaseUrl.replace(/\/+$/, "");
  // Public dashboard reads stay same-origin in the browser. The dev server
  // proxies /public to the coordinator, avoiding a cross-origin preflight;
  // production serves the same route from its own origin.
  if (path.startsWith("/public/v1/")) return path;
  if (!base.endsWith("/v1")) return `${base}${path}`;

  return `${base}${path.startsWith("/v1/") ? path.slice("/v1".length) : path}`;
}

export async function loadApiUsage(accessToken: string, limit = 50, apiBaseUrl?: string): Promise<ApiUsage[]> {
  const response = await apiRequest<{ data: ApiUsage[] }>(`/v1/account/usage?limit=${limit}`, accessToken, {}, apiBaseUrl);
  return response.data;
}

export function loadOperatorEvidence(accessToken: string, apiBaseUrl?: string): Promise<OperatorEvidenceSnapshot> {
  return apiRequest<OperatorEvidenceSnapshot>("/public/v1/admin/operations", accessToken, {}, apiBaseUrl);
}

export async function loadApiKeys(accessToken: string, apiBaseUrl?: string): Promise<ApiKeySummary[]> {
  const response = await apiRequest<{ data: ApiKeySummary[] }>("/v1/api-keys", accessToken, {}, apiBaseUrl);
  return response.data;
}

export function createApiKey(accessToken: string, name: string, apiBaseUrl?: string): Promise<CreatedApiKey> {
  return apiRequest<CreatedApiKey>("/v1/api-keys", accessToken, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  }, apiBaseUrl);
}

export async function revokeApiKey(accessToken: string, keyId: string, apiBaseUrl?: string): Promise<void> {
  await apiRequest<void>(`/v1/api-keys/${encodeURIComponent(keyId)}`, accessToken, {
    method: "DELETE",
  }, apiBaseUrl);
}

export async function loadOwnedNodes(accessToken: string, apiBaseUrl?: string): Promise<OwnedNode[]> {
  return (await apiRequest<{ data: OwnedNode[] }>("/v1/nodes", accessToken, {}, apiBaseUrl)).data;
}

export function enqueueOwnedNodeCommand(
  accessToken: string,
  accountId: string,
  node: Pick<OwnedNode, "nodeId" | "generation">,
  command: Pick<NodeCommand, "type" | "payload">,
): Promise<{ command: NodeCommand; state: string }> {
  const now = Date.now();
  const input = {
    schema: "mycellios-node-command/1",
    id: crypto.randomUUID(),
    nodeId: node.nodeId,
    actor: { kind: "account", id: accountId, scopes: commandScopes(command.type) },
    generation: node.generation,
    issuedAt: new Date(now).toISOString(),
    expiresAt: new Date(now + 5 * 60_000).toISOString(),
    nonce: `nonce_${crypto.randomUUID().replaceAll("-", "")}`,
    type: command.type,
    payload: command.payload,
  } as NodeCommand;
  return apiRequest(`/v1/nodes/${encodeURIComponent(node.nodeId)}/commands`, accessToken, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(input),
  });
}

export function revokeOwnedNode(
  accessToken: string,
  node: Pick<OwnedNode, "nodeId" | "credentialFingerprint">,
  reason: string,
  confirmation: string,
): Promise<{ state: string; identityId: string; generation: number; disconnected: number; affectedLeases: number }> {
  return apiRequest(`/v1/nodes/${encodeURIComponent(node.nodeId)}/revoke`, accessToken, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ expectedFingerprint: node.credentialFingerprint, reason, confirmation }),
  });
}

export function createNodeOwnershipTransfer(
  accessToken: string,
  node: Pick<OwnedNode, "nodeId" | "generation">,
  targetAccountId: string,
  confirmation: string,
): Promise<{ transferId: string; transferToken: string; nodeId: string; targetAccountId: string; expiresAt: string }> {
  return apiRequest(`/v1/nodes/${encodeURIComponent(node.nodeId)}/ownership-transfers`, accessToken, {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ targetAccountId, expectedGeneration: node.generation, confirmation, expiresInSeconds: 900 }),
  });
}

export function acceptNodeOwnershipTransfer(
  accessToken: string,
  input: { transferId: string; transferToken: string; nodeId: string; confirmation: string },
): Promise<{ state: "transferred"; nodeId: string; generation: number }> {
  return apiRequest(`/v1/node-ownership-transfers/${encodeURIComponent(input.transferId)}/accept`, accessToken, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({
      transferToken: input.transferToken, nodeId: input.nodeId, confirmation: input.confirmation,
    }),
  });
}

export async function createNodeEnrollmentBundle(
  accessToken: string,
  accountId: string,
  coordinatorUrl: string,
): Promise<NodeEnrollmentBundle> {
  const issued = await apiRequest<{ enrollmentId: string; enrollmentToken: string; nonce: string; expiresAt: string }>(
    "/v1/nodes/enrollments", accessToken, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ schema: "mycellios-node-enrollment-create/1", accountId,
        requestedBy: { kind: "account", id: accountId, scopes: ["node:identity"] }, expiresInSeconds: 900 }),
    },
  );
  await apiRequest<void>(`/v1/nodes/enrollments/${encodeURIComponent(issued.enrollmentId)}/confirm`, accessToken, { method: "POST" });
  return { schema: "mycellios-node-enrollment-bundle/1", coordinatorUrl: new URL(coordinatorUrl).origin,
    enrollmentId: issued.enrollmentId, enrollmentToken: issued.enrollmentToken, nonce: issued.nonce, expiresAt: issued.expiresAt };
}

function commandScopes(type: NodeCommand["type"]): Array<"node:control" | "node:limits" | "node:update" | "node:identity"> {
  if (type === "set-limits" || type === "set-policy") return ["node:limits"];
  if (type === "update" || type === "rollback") return ["node:update"];
  if (type === "revoke" || type === "uninstall") return ["node:identity"];
  return ["node:control"];
}

async function apiRequest<T>(
  path: string,
  accessToken: string,
  init: RequestInit = {},
  apiBaseUrl?: string,
): Promise<T> {
  const response = await fetch(apiRequestUrl(path, apiBaseUrl), {
    ...init,
    cache: "no-store",
    headers: {
      ...init.headers,
      authorization: `Bearer ${accessToken}`,
    },
  });
  if (response.status === 204) return undefined as T;
  const payload = await response.json().catch(() => null) as {
    error?: { code?: string; message?: string };
  } | null;
  if (!response.ok) {
    throw new Error(payload?.error?.message ?? payload?.error?.code ?? `API request failed (HTTP ${response.status}).`);
  }
  return payload as T;
}
import type { NodeCommand, NodeDesiredState, NodeEnrollmentBundle, NodeSnapshot } from "../../src/contracts/node-control";
import type { ExecutionReceipt } from "../../src/contracts/execution-receipt";
import type { ExecutionTopology } from "../../src/contracts/execution-topology";

export function loadExecutionReceipt(accessToken: string, jobId: string): Promise<ExecutionReceipt> {
  return apiRequest<ExecutionReceipt>(`/v1/requests/${encodeURIComponent(jobId)}/receipt`, accessToken);
}

export function loadExecutionTopology(accessToken: string, jobId: string): Promise<ExecutionTopology> {
  return apiRequest<ExecutionTopology>(`/v1/requests/${encodeURIComponent(jobId)}/topology`, accessToken);
}
