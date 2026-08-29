import {
  ArrowUpRight,
  Bot,
  Check,
  CircleAlert,
  LoaderCircle,
  MessageCircle,
  Send,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type {
  ChatStreamUpdate,
  SupportAssistantPublicConfig,
} from "../../src/contracts/dashboard";
import { consumeChatCompletionStreamWithRecovery } from "../../src/core/chat-stream";
import "./support-assistant.css";

interface AssistantMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

type AssistantDestination = "models" | "contribute" | "inference";

interface SupportAssistantProps {
  apiOrigin?: string;
  surface?: "landing" | "panel";
  onNavigate?: (destination: AssistantDestination) => void;
}

interface AssistantAction {
  key: string;
  title: string;
  detail: string;
  label: string;
  execute: () => Promise<void> | void;
}

const DEFAULT_WELCOME = "Hi, I’m the mycellios assistant. I can help with the network, models, installation, drivers, and device performance.";

function assistantSessionId(): string {
  if (typeof window === "undefined") return "server-render";
  const key = "mycellios.support-session";
  const saved = window.sessionStorage.getItem(key);
  if (saved) return saved;
  const created = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  window.sessionStorage.setItem(key, created);
  return created;
}

function messageId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function normalizeIntent(value: string): string {
  return value.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase();
}

function readError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

async function responseError(response: Response): Promise<Error> {
  const body = await response.json().catch(() => null) as { error?: { message?: string } } | null;
  return new Error(body?.error?.message ?? `HTTP ${response.status}`);
}

export function SupportAssistant({
  apiOrigin = "",
  surface = "landing",
  onNavigate,
}: SupportAssistantProps) {
  const [open, setOpen] = useState(false);
  const [config, setConfig] = useState<SupportAssistantPublicConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState(false);
  const [actionDone, setActionDone] = useState<string | null>(null);
  const [latestIntent, setLatestIntent] = useState("");
  const conversationRef = useRef<HTMLDivElement>(null);
  const sessionId = useMemo(assistantSessionId, []);

  async function loadConfig() {
    try {
      const nextConfig = await fetch(`${apiOrigin}/public/v1/assistant/config`, { cache: "no-store" }).then(async (response) => {
        if (!response.ok) throw await responseError(response);
        return response.json() as Promise<SupportAssistantPublicConfig>;
      });
      setConfig(nextConfig);
      setConfigError(null);
    } catch (caught) {
      setConfigError(readError(caught));
    }
  }

  useEffect(() => {
    void loadConfig();
  }, [apiOrigin]);

  useEffect(() => {
    if (!open) return;
    conversationRef.current?.scrollTo({ top: conversationRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, open, busy]);

  function navigate(destination: AssistantDestination) {
    if (onNavigate) {
      onNavigate(destination);
      setOpen(false);
      return;
    }
    const base = window.location.pathname === "/dashboard" ? "/dashboard" : "/network";
    window.location.assign(`${base}?view=${destination}`);
  }

  const pendingAction = useMemo<AssistantAction | null>(() => {
    if (!latestIntent || !config) return null;
    const intent = normalizeIntent(latestIntent);
    const requestsComputeChange = /\b(change|switch|set|use|enable|activate|adjust|configure|cambia|cambiar|pon|poner|usa|usar|activa|activar|ajusta|ajustar|configura|configurar)\b/.test(intent)
      && /\b(cpu|gpu|automatico|automatic|equilibrado|balanceado)\b/.test(intent);
    if (requestsComputeChange && config.allowDeviceControl) {
      return {
        key: "open-compute-controls",
        title: "Configure this device",
        detail: "Open the native node controls to review and confirm the resource policy.",
        label: "Open this device",
        execute: () => navigate("contribute"),
      };
    }

    if (/\b(modelo|modelos)\b/.test(intent) && /\b(conecta|conectar|activa|activar|disponible|disponibles|cargar|carga)\b/.test(intent)) {
      return {
        key: "open-models",
        title: "Open network models",
        detail: "Shared activation requires administrator authorization.",
        label: "View models",
        execute: () => navigate("models"),
      };
    }
    if (/\b(contribuir|contribution|conectar este|anadir este|añadir este|aportar)\b/.test(intent)) {
      return {
        key: "open-contribute",
        title: "Connect this device",
        detail: "Open the local controls to review which resources will be offered before connecting.",
        label: "Open contribution settings",
        execute: () => navigate("contribute"),
      };
    }
    return null;
  }, [config, latestIntent, onNavigate]);

  async function submitPrompt(event?: FormEvent, suggestion?: string) {
    event?.preventDefault();
    const prompt = (suggestion ?? draft).trim();
    if (!prompt || busy || !config?.available) return;
    const userMessage: AssistantMessage = { id: messageId("user"), role: "user", content: prompt };
    const assistantId = messageId("assistant");
    const requestMessages = [...messages, userMessage]
      .filter((message) => message.content.trim())
      .slice(-19)
      .map(({ role, content }) => ({ role, content }));
    setMessages((current) => [...current, userMessage, { id: assistantId, role: "assistant", content: "", streaming: true }]);
    setDraft("");
    setBusy(true);
    setError(null);
    setLatestIntent(prompt);
    setActionDone(null);
    try {
      const page = typeof window === "undefined" ? surface : `${surface}:${window.location.pathname}${window.location.search}`;
      const platform = typeof navigator === "undefined" ? "unknown" : navigator.userAgent.slice(0, 220);
      const onUpdate = (update: ChatStreamUpdate) => {
          setMessages((current) => current.map((message) =>
            message.id === assistantId
              ? { ...message, content: update.text, streaming: true }
              : message
          ));
        };
      const response = await consumeChatCompletionStreamWithRecovery(
            (_attempt, signal) => fetch(`${apiOrigin}/public/v1/assistant/chat`, {
              method: "POST",
              headers: { accept: "text/event-stream", "content-type": "application/json" },
              body: JSON.stringify({
                session_id: sessionId,
                messages: requestMessages,
                page,
                platform,
              }),
              signal,
            }),
            config.selectedModel ?? "mycellios-network",
            onUpdate,
            {
              sessionId,
              maximumAttempts: 4,
              retryDelayMs: 900,
              connectionTimeoutMs: 15_000,
              streamIdleTimeoutMs: 35_000,
            },
          );
      setMessages((current) => current.map((message) =>
        message.id === assistantId
          ? { ...message, content: response.text, streaming: false }
          : message
      ));
    } catch (caught) {
      setMessages((current) => current.filter((message) => message.id !== assistantId));
      setError(readError(caught));
      void loadConfig();
    } finally {
      setBusy(false);
    }
  }

  async function executeAction(action: AssistantAction) {
    setActionBusy(true);
    setError(null);
    try {
      await action.execute();
      setActionDone(action.key);
    } catch (caught) {
      setError(readError(caught));
    } finally {
      setActionBusy(false);
    }
  }

  const online = config?.enabled && config.available;
  const welcome = config?.welcomeMessage ?? DEFAULT_WELCOME;

  return (
    <div className={`support-assistant support-assistant-${surface}`}>
      {open && (
        <section className="support-assistant-window" role="dialog" aria-label="mycellios assistant">
          <header>
            <div className="support-assistant-avatar"><Bot size={20} /></div>
            <div className="support-assistant-title">
              <strong>mycellios assistant</strong>
              <i
                className={`support-assistant-connection${online ? " online" : " offline"}`}
                role="status"
                aria-label={online ? "Assistant connected" : "Assistant disconnected"}
                title={online ? "Connected" : "Disconnected"}
              />
            </div>
            <button type="button" title="Close" aria-label="Close assistant" onClick={() => setOpen(false)}><X size={18} /></button>
          </header>

          <div className="support-assistant-conversation" ref={conversationRef}>
            <div className="support-assistant-message assistant">
              <div className="support-assistant-mini-avatar"><Sparkles size={14} /></div>
              <p>{welcome}</p>
            </div>

            {messages.map((message) => (
              <div className={`support-assistant-message ${message.role}`} key={message.id}>
                {message.role === "assistant" && <div className="support-assistant-mini-avatar"><Sparkles size={14} /></div>}
                <p>{message.content || (message.streaming ? "Connecting to the network model…" : "")}{message.streaming && message.content && <i className="support-assistant-caret" />}</p>
              </div>
            ))}

            {messages.length === 0 && config?.suggestions && (
              <div className="support-assistant-suggestions">
                {config.suggestions.map((suggestion) => (
                  <button type="button" key={suggestion} disabled={!online} onClick={() => void submitPrompt(undefined, suggestion)}>
                    {suggestion}<ArrowUpRight size={13} />
                  </button>
                ))}
              </div>
            )}

            {pendingAction && actionDone !== pendingAction.key && !busy && (
              <article className="support-assistant-action">
                <div><Zap size={17} /><span><strong>{pendingAction.title}</strong><small>{pendingAction.detail}</small></span></div>
                <button type="button" disabled={actionBusy} onClick={() => void executeAction(pendingAction)}>
                  {actionBusy ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}{pendingAction.label}
                </button>
              </article>
            )}

            {pendingAction && actionDone === pendingAction.key && (
              <div className="support-assistant-success"><Check size={14} /> Action applied on this device.</div>
            )}

            {(error || configError) && (
              <div className="support-assistant-error"><CircleAlert size={15} /><span>{error ?? "The assistant status could not be retrieved."}</span></div>
            )}

            {config && !config.enabled && (
              <div className="support-assistant-unavailable"><CircleAlert size={16} /><span><strong>Assistant disabled</strong><small>The network administrator has temporarily disabled it.</small></span></div>
            )}
          </div>

          <form className="support-assistant-composer" onSubmit={(event) => void submitPrompt(event)}>
            <textarea
              rows={2}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void submitPrompt();
                }
              }}
              placeholder={online ? "Ask about mycellios, your GPU, or the network…" : "Waiting for a network model…"}
              disabled={!online || busy}
              aria-label="Message for the assistant"
            />
            <button type="submit" disabled={!online || busy || !draft.trim()} aria-label="Send message">
              {busy ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}
            </button>
          </form>
        </section>
      )}

      <button
        className={`support-assistant-launcher ${open ? "open" : ""}`}
        type="button"
        aria-label={open ? "Close assistant" : "Open mycellios assistant"}
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        {open ? <X size={22} /> : <MessageCircle size={23} />}
        {!open && <span><strong>Need help?</strong><small>{online ? "Ask the network" : "Assistant offline"}</small></span>}
        {!open && <i
          className={online ? "online" : "offline"}
          role="status"
          aria-label={online ? "Assistant connected" : "Assistant disconnected"}
          title={online ? "Connected" : "Disconnected"}
        />}
      </button>
    </div>
  );
}
