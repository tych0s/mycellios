import {
  ArrowUpRight,
  Bot,
  Check,
  CircleAlert,
  Cpu,
  LoaderCircle,
  MessageCircle,
  RotateCcw,
  Send,
  ShieldCheck,
  Sparkles,
  X,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type {
  ChatStreamUpdate,
  DesktopBridge,
  DesktopSettings,
  SupportAssistantPublicConfig,
} from "../../src/desktop/contracts";
import { consumeChatCompletionStreamWithRecovery } from "../../src/desktop/chat-stream";
import "./support-assistant.css";

interface AssistantMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

type AssistantDestination = "models" | "contribute" | "inference" | "machine";

interface AssistantDeviceControl {
  currentMode: DesktopSettings["computeMode"];
  onChange: (mode: DesktopSettings["computeMode"]) => Promise<void>;
}

interface SupportAssistantProps {
  apiOrigin?: string;
  surface?: "landing" | "panel";
  onNavigate?: (destination: AssistantDestination) => void;
  deviceControl?: AssistantDeviceControl;
  desktopBridge?: DesktopBridge;
}

interface AssistantAction {
  key: string;
  title: string;
  detail: string;
  label: string;
  execute: () => Promise<void> | void;
}

const DEFAULT_WELCOME = "Hola, soy el asistente de mycellios. Puedo ayudarte con la red, modelos, instalación, drivers y potencia del dispositivo.";

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
  deviceControl,
  desktopBridge,
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
      const nextConfig = desktopBridge
        ? await desktopBridge.getSupportAssistantConfig()
        : await fetch(`${apiOrigin}/public/v1/assistant/config`, { cache: "no-store" }).then(async (response) => {
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
  }, [apiOrigin, desktopBridge]);

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
    const view = destination === "machine" ? "contribute" : destination;
    window.location.assign(`/network?view=${view}`);
  }

  const pendingAction = useMemo<AssistantAction | null>(() => {
    if (!latestIntent || !config) return null;
    const intent = normalizeIntent(latestIntent);
    const requestsChange = /\b(cambia|cambiar|pon|poner|usa|usar|activa|activar|quiero|ajusta|ajustar|configura|configurar)\b/.test(intent);
    const computeMode = intent.includes("cpu")
      ? "cpu-only"
      : /\b(gpu|maxima potencia|rendimiento maximo)\b/.test(intent)
        ? "gpu-only"
        : /\b(automatico|automatic|equilibrado|balanceado)\b/.test(intent)
          ? "automatic"
          : null;

    if (requestsChange && computeMode && config.allowDeviceControl) {
      const modeLabel = computeMode === "gpu-only"
        ? "Sólo GPU"
        : computeMode === "cpu-only"
          ? "Sólo CPU"
          : "Automático";
      if (deviceControl) {
        if (deviceControl.currentMode === computeMode) return null;
        return {
          key: `compute-${computeMode}`,
          title: `Cambiar este equipo a ${modeLabel}`,
          detail: "Esta acción sólo cambia el modo de cálculo local. No modifica otros nodos.",
          label: `Confirmar ${modeLabel}`,
          execute: () => deviceControl.onChange(computeMode),
        };
      }
      return {
        key: `open-compute-${computeMode}`,
        title: `Configurar ${modeLabel}`,
        detail: "Abre los controles de este dispositivo para que confirmes el cambio allí.",
        label: "Abrir este dispositivo",
        execute: () => navigate("machine"),
      };
    }

    if (/\b(modelo|modelos)\b/.test(intent) && /\b(conecta|conectar|activa|activar|disponible|disponibles|cargar|carga)\b/.test(intent)) {
      return {
        key: "open-models",
        title: "Abrir modelos de la red",
        detail: "La activación compartida requiere autorización de administrador.",
        label: "Ver modelos",
        execute: () => navigate("models"),
      };
    }
    if (/\b(contribuir|contribution|conectar este|anadir este|añadir este|aportar)\b/.test(intent)) {
      return {
        key: "open-contribute",
        title: "Conectar este dispositivo",
        detail: "Abre el control local para revisar qué recursos se ofrecerán antes de conectarlo.",
        label: "Abrir contribución",
        execute: () => navigate(deviceControl ? "machine" : "contribute"),
      };
    }
    return null;
  }, [config, deviceControl, latestIntent, onNavigate]);

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
      const response = desktopBridge
        ? await desktopBridge.streamSupportAssistant({
            sessionId,
            messages: requestMessages,
            page,
            platform,
          }, onUpdate)
        : await consumeChatCompletionStreamWithRecovery(
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

  function resetConversation() {
    setMessages([]);
    setDraft("");
    setError(null);
    setLatestIntent("");
    setActionDone(null);
  }

  const online = config?.enabled && config.available;
  const welcome = config?.welcomeMessage ?? DEFAULT_WELCOME;

  return (
    <div className={`support-assistant support-assistant-${surface}`}>
      {open && (
        <section className="support-assistant-window" role="dialog" aria-label="Asistente de mycellios">
          <header>
            <div className="support-assistant-avatar"><Bot size={20} /></div>
            <div>
              <strong>Asistente mycellios</strong>
              <span className={online ? "online" : "offline"}><i />{online ? "Modelo de la red conectado" : "Sin modelo de red"}</span>
            </div>
            <button type="button" title="Nueva conversación" aria-label="Nueva conversación" onClick={resetConversation}><RotateCcw size={16} /></button>
            <button type="button" title="Cerrar" aria-label="Cerrar asistente" onClick={() => setOpen(false)}><X size={18} /></button>
          </header>

          <div className="support-assistant-trust">
            <ShieldCheck size={14} />
            <span>IA de la red únicamente</span>
            <b>{config?.selectedModel ?? "ningún modelo"}</b>
          </div>

          <div className="support-assistant-conversation" ref={conversationRef}>
            <div className="support-assistant-message assistant">
              <div className="support-assistant-mini-avatar"><Sparkles size={14} /></div>
              <p>{welcome}</p>
            </div>

            {messages.map((message) => (
              <div className={`support-assistant-message ${message.role}`} key={message.id}>
                {message.role === "assistant" && <div className="support-assistant-mini-avatar"><Sparkles size={14} /></div>}
                <p>{message.content || (message.streaming ? "Conectando con el modelo de la red…" : "")}{message.streaming && message.content && <i className="support-assistant-caret" />}</p>
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
              <div className="support-assistant-success"><Check size={14} /> Acción aplicada en este dispositivo.</div>
            )}

            {(error || configError) && (
              <div className="support-assistant-error"><CircleAlert size={15} /><span>{error ?? "No se puede consultar el estado del asistente."}</span></div>
            )}

            {config && !config.enabled && (
              <div className="support-assistant-unavailable"><CircleAlert size={16} /><span><strong>Asistente desactivado</strong><small>El administrador de la red lo ha desactivado temporalmente.</small></span></div>
            )}
            {config?.enabled && !config.available && (
              <div className="support-assistant-unavailable"><Cpu size={16} /><span><strong>No hay un modelo real conectado</strong><small>El chat no usa una IA externa como sustituto. Volverá a estar disponible cuando la red anuncie un modelo.</small></span></div>
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
              placeholder={online ? "Pregunta sobre mycellios, tu GPU o la red…" : "Esperando un modelo de la red…"}
              disabled={!online || busy}
              aria-label="Mensaje para el asistente"
            />
            <button type="submit" disabled={!online || busy || !draft.trim()} aria-label="Enviar mensaje">
              {busy ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}
            </button>
          </form>
          <footer>Las acciones que cambian el sistema siempre requieren confirmación.</footer>
        </section>
      )}

      <button
        className={`support-assistant-launcher ${open ? "open" : ""}`}
        type="button"
        aria-label={open ? "Cerrar asistente" : "Abrir asistente de mycellios"}
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
      >
        {open ? <X size={22} /> : <MessageCircle size={23} />}
        {!open && <span><strong>¿Necesitas ayuda?</strong><small>{online ? "Pregunta a la red" : "Asistente sin conexión"}</small></span>}
        {!open && <i className={online ? "online" : ""} />}
      </button>
    </div>
  );
}
