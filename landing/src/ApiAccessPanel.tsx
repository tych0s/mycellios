import {
  Check,
  CircleAlert,
  Clipboard,
  Code2,
  KeyRound,
  LoaderCircle,
  LockKeyhole,
  PlugZap,
  Plus,
  Trash2,
  Wifi,
} from "lucide-react";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import {
  createApiKey,
  loadApiKeys,
  revokeApiKey,
  type ApiAccount,
  type ApiKeySummary,
  type CreatedApiKey,
} from "./api-access";
import type { AuthSession } from "./auth";
import { TABLE_PAGE_SIZE, TablePagination } from "./TablePagination";

interface ApiAccessPanelProps {
  enabled: boolean;
  apiBaseUrl: string;
  accessToken: string | null;
  getValidSession?: (forceRefresh?: boolean) => Promise<AuthSession | null>;
  account: ApiAccount | null;
  availableModels: number;
  onSignIn: () => void;
}

export function ApiAccessPanel({
  enabled,
  apiBaseUrl,
  accessToken,
  getValidSession,
  account,
  availableModels,
  onSignIn,
}: ApiAccessPanelProps) {
  const [open, setOpen] = useState(false);
  const [keys, setKeys] = useState<ApiKeySummary[]>([]);
  const [name, setName] = useState("My application");
  const [created, setCreated] = useState<CreatedApiKey | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<"idle" | "testing" | "ready" | "empty" | "failed">("idle");
  const [keysPage, setKeysPage] = useState(0);
  const keysPageSize = TABLE_PAGE_SIZE;
  const keysPageCount = Math.max(1, Math.ceil(keys.length / keysPageSize));
  const visibleKeysPage = Math.min(keysPage, keysPageCount - 1);
  const visibleKeys = keys.slice(visibleKeysPage * keysPageSize, visibleKeysPage * keysPageSize + keysPageSize);
  const normalizedBase = apiBaseUrl.replace(/\/+$/, "");
  const origin = normalizedBase.replace(/\/v1$/, "");
  const commands = useMemo(() => [
    {
      label: "API ENDPOINT",
      value: normalizedBase,
      help: "OpenAI-compatible endpoint ready to configure in a client.",
    },
    {
      label: "EXPORT BASE URL",
      value: `$env:OPENAI_BASE_URL=\"${normalizedBase}\"`,
      help: "Environment variable for PowerShell.",
    },
    {
      label: "LIST MODELS COMMAND",
      value: `curl.exe ${normalizedBase}/models -H \"Authorization: Bearer $env:MYCELLIOS_API_KEY\"`,
      help: "Checks the endpoint and returns only models that are actually available.",
    },
  ], [normalizedBase]);

  async function withCurrentSession<T>(operation: (accessToken: string) => Promise<T>): Promise<T> {
    const session = getValidSession ? await getValidSession(false) : null;
    const token = getValidSession ? session?.accessToken : accessToken;
    if (!token) throw new Error("Sign in to continue.");
    try {
      return await operation(token);
    } catch (caught) {
      if (!getValidSession || !apiAccessSessionNeedsRefresh(caught)) throw caught;
      const renewed = await getValidSession(true);
      if (!renewed) throw caught;
      return operation(renewed.accessToken);
    }
  }

  useEffect(() => {
    if (!open || !accessToken) return;
    let cancelled = false;
    setBusy(true);
    setError(null);
    void withCurrentSession((token) => loadApiKeys(token, apiBaseUrl))
      .then((next) => { if (!cancelled) setKeys(next); })
      .catch((caught) => { if (!cancelled) setError(errorText(caught)); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [accessToken, open]);

  useEffect(() => setKeysPage(0), [keys.length]);

  async function testConnection() {
    if (!accessToken) {
      onSignIn();
      return;
    }
    setConnection("testing");
    setError(null);
    try {
      const payload = await withCurrentSession(async (token) => {
        const response = await fetch(`${normalizedBase}/models`, {
          cache: "no-store",
          headers: { authorization: `Bearer ${token}` },
        });
        const body = await response.json().catch(() => null) as {
          data?: unknown[];
          error?: { code?: string; message?: string };
        } | null;
        if (!response.ok) throw new Error(body?.error?.code ?? body?.error?.message ?? `HTTP ${response.status}`);
        return body;
      });
      setConnection((payload?.data?.length ?? 0) > 0 ? "ready" : "empty");
    } catch (caught) {
      setConnection("failed");
      setError(errorText(caught));
    }
  }

  async function addKey(event: FormEvent) {
    event.preventDefault();
    if (!accessToken || !name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const next = await withCurrentSession((token) => createApiKey(token, name.trim(), apiBaseUrl));
      setCreated(next);
      setKeys((current) => [next, ...current]);
      setName("My application");
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  }

  async function removeKey(keyId: string) {
    if (!accessToken) return;
    setBusy(true);
    setError(null);
    try {
      await withCurrentSession((token) => revokeApiKey(token, keyId, apiBaseUrl));
      setKeys((current) => current.map((key) =>
        key.id === keyId ? { ...key, revoked_at: new Date().toISOString() } : key
      ));
    } catch (caught) {
      setError(errorText(caught));
    } finally {
      setBusy(false);
    }
  }

  const live = availableModels > 0;
  const targetLabel = live ? "API TARGET READY" : "API TARGET WAITING";
  const errorPresentation = error ? apiAccessErrorPresentation(error) : null;
  return <section className={`api-access-panel${open ? " open" : ""}`}>
    <button className="api-access-summary" type="button" onClick={() => setOpen((current) => !current)} aria-expanded={open}>
      <span className={`api-target-state${live ? " ready" : ""}`}><i />{targetLabel}</span>
      <strong>{origin}</strong>
      <span className="api-access-summary-copy"><Code2 size={15} />API Access</span>
      <span className="api-access-chevron">{open ? "−" : "+"}</span>
    </button>
    {open && <div className="api-access-body">
      <header>
        <div className="api-access-icon"><PlugZap /></div>
        <div><span>OPENAI-COMPATIBLE</span><h2>Connect mycellios to your applications</h2><p>Use the same client format while requests run on the mycellios network.</p></div>
      </header>

      {!enabled && <div className="api-access-notice"><CircleAlert /><span><strong>Account access is not configured</strong>The local coordinator keeps the API available, but public keys require the account service.</span></div>}
      {enabled && !accessToken && <div className="api-access-notice signin"><LockKeyhole /><span><strong>Sign in to get a key</strong>The endpoint is visible, but access and balance belong to your account.</span><button onClick={onSignIn}>Sign in</button></div>}

      <div className="api-command-grid">
        {commands.map((command) => <div className="api-command" key={command.label}>
          <div><span>{command.label}</span><CopyButton value={command.value} /></div>
          <code>{command.value}</code>
          <p>{command.help}</p>
        </div>)}
      </div>

      <div className="api-access-status-row">
        <div>
          <span>LIVE STATUS</span>
          <strong className={live ? "ready" : ""}>{live ? `${availableModels} model${availableModels === 1 ? "" : "s"} available` : "No models connected"}</strong>
          <small>The API is not marked operational until `/models` returns real capacity.</small>
        </div>
        <button className="secondary-button" disabled={connection === "testing"} onClick={() => void testConnection()}>
          {connection === "testing" ? <LoaderCircle className="spin" /> : connection === "ready" ? <Check /> : <Wifi />}
          {connection === "idle" ? "Test connection" : connection === "testing" ? "Testing…" : connection === "ready" ? "Connection ready" : connection === "empty" ? "API active, no model" : "Try again"}
        </button>
      </div>

      {accessToken && account && <div className="api-account-strip">
        <div><span>BALANCE</span><strong>{formatTokens(account.token_balance)} TOK</strong></div>
        <div><span>USED</span><strong>{formatTokens(account.lifetime_input_tokens + account.lifetime_output_tokens)}</strong></div>
        <div><span>REQUESTS</span><strong>{account.request_count}</strong></div>
        <div><span>LIMITS</span><strong>{account.limits.requests_per_minute}/min · {account.limits.max_concurrent} concurrent</strong></div>
      </div>}

      {accessToken && <section className="api-keys-section">
        <div className="api-keys-heading"><div><span>API KEYS</span><h3>Revocable keys</h3></div><small>The full key is shown only once.</small></div>
        {created && <div className="api-created-key"><div><KeyRound /><span><strong>Save it now</strong><code>{created.secret}</code></span></div><CopyButton value={created.secret} label="Copy key" /></div>}
        <form onSubmit={(event) => void addKey(event)} className="api-key-form">
          <input value={name} onChange={(event) => setName(event.target.value)} maxLength={80} aria-label="API key name" placeholder="Application name" />
          <button className="primary-button" disabled={busy || !name.trim()}><Plus />Create key</button>
        </form>
        <div className="api-key-list">
          {busy && keys.length === 0 && <div className="api-key-empty"><LoaderCircle className="spin" />Loading keys…</div>}
          {!busy && keys.length === 0 && <div className="api-key-empty"><KeyRound />You do not have any keys yet.</div>}
          {visibleKeys.map((key) => <div className={`api-key-row${key.revoked_at ? " revoked" : ""}`} key={key.id}>
            <KeyRound />
            <span><strong>{key.name}</strong><code>{key.prefix}••••••••</code></span>
            <small>{key.revoked_at ? "Revoked" : key.last_used_at ? `Used ${relativeDate(key.last_used_at)}` : "Never used"}</small>
            {!key.revoked_at && <button aria-label={`Revoke ${key.name}`} disabled={busy} onClick={() => void removeKey(key.id)}><Trash2 /></button>}
          </div>)}
        </div>
        {keys.length > keysPageSize && <TablePagination label="API keys" page={visibleKeysPage} pageCount={keysPageCount} total={keys.length} onPageChange={setKeysPage} />}
      </section>}
      {errorPresentation && <div className="api-access-error" role="alert">
        <CircleAlert />
        <span><strong>{errorPresentation.title}</strong><small>{errorPresentation.detail}</small></span>
        <button type="button" onClick={() => { setError(null); if (errorPresentation.requiresAuth) onSignIn(); else void testConnection(); }}>
          {errorPresentation.action}
        </button>
      </div>}
    </div>}
  </section>;
}

function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    await navigator.clipboard.writeText(value);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1_500);
  }
  return <button type="button" className="api-copy-button" onClick={() => void copy()}>{copied ? <Check /> : <Clipboard />}{copied ? "Copied" : label}</button>;
}

function formatTokens(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
}

function relativeDate(value: string): string {
  const milliseconds = Date.now() - new Date(value).getTime();
  if (milliseconds < 60_000) return "just now";
  if (milliseconds < 3_600_000) return `${Math.floor(milliseconds / 60_000)} min ago`;
  if (milliseconds < 86_400_000) return `${Math.floor(milliseconds / 3_600_000)} h ago`;
  return new Date(value).toLocaleDateString("en-US");
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function apiAccessSessionNeedsRefresh(error: unknown): boolean {
  const message = errorText(error).toLowerCase();
  return message.includes("invalid_network_token")
    || message.includes("invalid network token")
    || message.includes("invalid_access_token")
    || message.includes("account session is invalid")
    || message.includes("session is invalid or expired");
}

export interface ApiAccessErrorPresentation {
  title: string;
  detail: string;
  action: string;
  requiresAuth: boolean;
}

export function apiAccessErrorPresentation(message: string): ApiAccessErrorPresentation {
  const normalized = message.toLowerCase();
  if (normalized.includes("invalid_network_token") || normalized.includes("invalid network token")) {
    return {
      title: "La sesión de API ha caducado",
      detail: "La credencial de esta sesión ya no es válida. Inicia sesión de nuevo para comprobar la conexión y administrar tus claves.",
      action: "Iniciar sesión",
      requiresAuth: true,
    };
  }
  return {
    title: "No se pudo comprobar la API",
    detail: message || "El endpoint no ha podido confirmar su estado.",
    action: "Reintentar",
    requiresAuth: false,
  };
}
