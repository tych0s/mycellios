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

export async function loadApiAccount(accessToken: string): Promise<ApiAccount> {
  return apiRequest<ApiAccount>("/v1/account", accessToken);
}

export async function loadApiKeys(accessToken: string): Promise<ApiKeySummary[]> {
  const response = await apiRequest<{ data: ApiKeySummary[] }>("/v1/api-keys", accessToken);
  return response.data;
}

export function createApiKey(accessToken: string, name: string): Promise<CreatedApiKey> {
  return apiRequest<CreatedApiKey>("/v1/api-keys", accessToken, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function revokeApiKey(accessToken: string, keyId: string): Promise<void> {
  await apiRequest<void>(`/v1/api-keys/${encodeURIComponent(keyId)}`, accessToken, {
    method: "DELETE",
  });
}

async function apiRequest<T>(
  path: string,
  accessToken: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: {
      ...init.headers,
      authorization: `Bearer ${accessToken}`,
    },
  });
  if (response.status === 204) return undefined as T;
  const payload = await response.json().catch(() => null) as {
    error?: { message?: string };
  } | null;
  if (!response.ok) {
    throw new Error(payload?.error?.message ?? `API request failed (HTTP ${response.status}).`);
  }
  return payload as T;
}
