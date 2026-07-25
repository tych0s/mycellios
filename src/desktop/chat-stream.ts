import type { ChatResponse, ChatStreamUpdate } from "./contracts.js";

interface MycelliosStreamChunk {
  id?: string;
  model?: string;
  choices?: Array<{
    delta?: { content?: string };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  x_network?: {
    session_id?: string;
    route_class?: string;
    affinity_hit?: boolean;
    reused_kv_tokens?: number;
    token_index?: number;
    ttft_ms?: number;
    active_ms?: number;
    phase?: ChatStreamUpdate["phase"];
    status_message?: string;
    attempt?: number;
    affected_worker_id?: string;
    affected_node_id?: string;
  };
  error?: { code?: string; message?: string };
}

const RECOVERABLE_PRETOKEN_CODES = new Set([
  "connection_timeout",
  "first_token_timeout",
  "pipeline_stage_disconnected",
  "stream_idle_timeout",
  "worker_disconnected",
  "worker_unreachable",
  "lease_accept_timeout",
  "no_capacity",
]);

export class ChatStreamError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ChatStreamError";
  }
}

export interface RecoveringChatStreamOptions {
  sessionId: string;
  maximumAttempts?: number;
  retryDelayMs?: number;
  connectionTimeoutMs?: number;
  streamIdleTimeoutMs?: number;
}

export async function consumeChatCompletionStreamWithRecovery(
  openResponse: (attempt: number, signal: AbortSignal) => Promise<Response>,
  fallbackModel: string,
  onUpdate: (update: ChatStreamUpdate) => void,
  options: RecoveringChatStreamOptions,
): Promise<ChatResponse> {
  const maximumAttempts = Math.max(1, Math.min(12, options.maximumAttempts ?? 8));
  const retryDelayMs = Math.max(0, options.retryDelayMs ?? 1_000);
  const connectionTimeoutMs = Math.max(1, options.connectionTimeoutMs ?? 15_000);
  const streamIdleTimeoutMs = Math.max(1, options.streamIdleTimeoutMs ?? 25_000);
  let latest: ChatStreamUpdate = {
    requestId: "pending",
    model: fallbackModel,
    delta: "",
    text: "",
    outputTokens: 0,
    routeClass: "waiting",
    affinityHit: false,
    sessionId: options.sessionId,
    reusedKvTokens: 0,
    ttftMs: 0,
    elapsedMs: 0,
    phase: "connecting",
    statusMessage: "Connecting to an available model route.",
    attempt: 1,
    maximumAttempts,
  };
  let replayPrefix = "";
  onUpdate(latest);
  for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
    const startedAt = Date.now();
    const attemptController = new AbortController();
    let connectionTimedOut = false;
    const connectionTimer = setTimeout(() => {
      connectionTimedOut = true;
      attemptController.abort();
    }, connectionTimeoutMs);
    try {
      const response = await openResponse(attempt, attemptController.signal);
      clearTimeout(connectionTimer);
      return await consumeChatCompletionStream(
        response,
        fallbackModel,
        (update) => {
          const replaying = replayPrefix.length > 0 && replayPrefix.startsWith(update.text);
          if (!replaying) replayPrefix = "";
          latest = {
            ...update,
            ...(replaying
              ? {
                  delta: "",
                  text: replayPrefix,
                  outputTokens: Math.max(latest.outputTokens, update.outputTokens),
                  phase: "recovering" as const,
                  statusMessage: "Reconectado. Recuperando el punto alcanzado antes del corte…",
                }
              : update.outputTokens > 0
              ? { phase: "streaming" as const }
              : update.phase
                ? { phase: update.phase }
                : {}),
            attempt,
            maximumAttempts,
          };
          onUpdate(latest);
        },
        startedAt,
        streamIdleTimeoutMs,
        () => attemptController.abort(),
      );
    } catch (caught) {
      const error = connectionTimedOut
        ? new ChatStreamError(
            "connection_timeout",
            "The coordinator connection stopped responding while the network was changing.",
          )
        : caught;
      if (
        attempt >= maximumAttempts ||
        !isRecoverablePretokenStreamError(error)
      ) {
        throw error;
      }
      if (latest.text) replayPrefix = latest.text;
      latest = {
        ...latest,
        requestId: "pending",
        phase: "recovering",
        statusMessage: connectionTimedOut
          ? "La conexión cambió o dejó de responder. Reconectando automáticamente…"
          : "La ruta se interrumpió antes del primer token. Reconectando automáticamente…",
        attempt: attempt + 1,
        maximumAttempts,
        elapsedMs: Math.max(latest.elapsedMs, Date.now() - startedAt),
      };
      onUpdate(latest);
      await waitForReconnect(Math.min(8_000, retryDelayMs * (2 ** (attempt - 1))));
    } finally {
      clearTimeout(connectionTimer);
    }
  }
  throw new ChatStreamError("retry_exhausted", "The inference retry was exhausted.");
}

export async function consumeChatCompletionStream(
  response: Response,
  fallbackModel: string,
  onUpdate: (update: ChatStreamUpdate) => void,
  startedAt = Date.now(),
  streamIdleTimeoutMs = 25_000,
  abortAttempt?: () => void,
): Promise<ChatResponse> {
  if (!response.ok) throw await responseError(response);
  if (!response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new Error("El coordinador no devolvió un flujo de tokens válido.");
  }
  if (!response.body) throw new Error("El coordinador terminó sin abrir el flujo de tokens.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let requestId = response.headers.get("x-network-request-id") ?? "unknown";
  let model = fallbackModel;
  let text = "";
  let outputTokens = 0;
  let promptTokens = 0;
  let totalTokens = 0;
  let routeClass = "unknown";
  let affinityHit = false;
  let sessionId = "";
  let reusedKvTokens = 0;
  let ttftMs = 0;
  let activeMs = 0;
  let completed = false;
  let doneMarker = false;

  const emit = (
    delta: string,
    extra: Partial<Pick<
      ChatStreamUpdate,
      "phase" | "statusMessage" | "attempt" | "affectedWorkerId" | "affectedNodeId"
    >> = {},
  ) => onUpdate({
    requestId,
    model,
    delta,
    text,
    outputTokens,
    routeClass,
    affinityHit,
    sessionId,
    reusedKvTokens,
    ttftMs,
    elapsedMs: Math.max(0, Date.now() - startedAt),
    ...extra,
  });

  const consumeEvent = (event: string) => {
    const data = event
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data) return;
    if (data.trim() === "[DONE]") {
      doneMarker = true;
      return;
    }

    let chunk: MycelliosStreamChunk;
    try {
      chunk = JSON.parse(data) as MycelliosStreamChunk;
    } catch {
      throw new Error("El coordinador devolvió un fragmento de tokens no válido.");
    }
    if (chunk.error?.message) {
      throw new ChatStreamError(chunk.error.code ?? "stream_error", chunk.error.message);
    }
    if (chunk.id) requestId = chunk.id;
    if (chunk.model) model = chunk.model;
    const receivedRoute = chunk.x_network?.route_class !== undefined;
    if (chunk.x_network?.route_class) routeClass = chunk.x_network.route_class;
    if (chunk.x_network?.affinity_hit !== undefined) affinityHit = chunk.x_network.affinity_hit;
    if (chunk.x_network?.session_id) sessionId = chunk.x_network.session_id;
    if (chunk.x_network?.reused_kv_tokens !== undefined) {
      reusedKvTokens = chunk.x_network.reused_kv_tokens;
    }
    const phase = chunk.x_network?.phase;
    const statusMessage = chunk.x_network?.status_message;
    const attempt = chunk.x_network?.attempt;
    const affectedWorkerId = chunk.x_network?.affected_worker_id;
    const affectedNodeId = chunk.x_network?.affected_node_id;

    const delta = chunk.choices?.[0]?.delta?.content ?? "";
    if (delta) {
      text += delta;
      outputTokens = chunk.x_network?.token_index === undefined
        ? outputTokens + 1
        : Math.max(outputTokens, chunk.x_network.token_index + 1);
      if (ttftMs === 0) ttftMs = Math.max(1, Date.now() - startedAt);
      emit(delta, {
        phase: "streaming",
        ...(statusMessage ? { statusMessage } : {}),
        ...(attempt === undefined ? {} : { attempt }),
        ...(affectedWorkerId ? { affectedWorkerId } : {}),
        ...(affectedNodeId ? { affectedNodeId } : {}),
      });
    }

    if (chunk.usage) {
      promptTokens = chunk.usage.prompt_tokens ?? promptTokens;
      outputTokens = chunk.usage.completion_tokens ?? outputTokens;
      totalTokens = chunk.usage.total_tokens ?? promptTokens + outputTokens;
      ttftMs = chunk.x_network?.ttft_ms ?? ttftMs;
      activeMs = chunk.x_network?.active_ms ?? Math.max(0, Date.now() - startedAt);
      completed = true;
      emit("");
    } else if (!delta && (receivedRoute || phase !== undefined || statusMessage !== undefined)) {
      emit("", {
        ...(phase ? { phase } : {}),
        ...(statusMessage ? { statusMessage } : {}),
        ...(attempt === undefined ? {} : { attempt }),
        ...(affectedWorkerId ? { affectedWorkerId } : {}),
        ...(affectedNodeId ? { affectedNodeId } : {}),
      });
    }
  };

  while (!doneMarker) {
    const { done, value } = await readStreamChunk(reader, streamIdleTimeoutMs, abortAttempt);
    buffer += decoder.decode(value, { stream: !done }).replaceAll("\r\n", "\n");
    let boundary = buffer.indexOf("\n\n");
    while (boundary >= 0) {
      const event = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      consumeEvent(event);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }

  if (!completed) throw new Error("El flujo terminó antes de recibir el resultado completo.");
  return {
    requestId,
    model,
    text,
    promptTokens,
    outputTokens,
    totalTokens: totalTokens || promptTokens + outputTokens,
    routeClass,
    affinityHit,
    sessionId,
    reusedKvTokens,
    ttftMs,
    activeMs,
  };
}

export function isRecoverablePretokenStreamError(error: unknown): boolean {
  if (error instanceof ChatStreamError) return RECOVERABLE_PRETOKEN_CODES.has(error.code);
  if (error instanceof TypeError) return true;
  return typeof DOMException !== "undefined" && error instanceof DOMException &&
    (error.name === "AbortError" || error.name === "TimeoutError" || error.name === "NetworkError");
}

async function readStreamChunk(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  timeoutMs: number,
  abortAttempt?: () => void,
) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      reader.read(),
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => {
          abortAttempt?.();
          void reader.cancel().catch(() => undefined);
          reject(new ChatStreamError(
            "stream_idle_timeout",
            "The token stream stopped responding after the network changed.",
          ));
        }, timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function waitForReconnect(milliseconds: number): Promise<void> {
  if (milliseconds <= 0) return Promise.resolve();
  return new Promise((resolve) => {
    const onlineEvents = globalThis as unknown as {
      addEventListener?: (type: string, listener: () => void, options?: { once?: boolean }) => void;
      removeEventListener?: (type: string, listener: () => void) => void;
    };
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      onlineEvents.removeEventListener?.("online", finish);
      resolve();
    };
    const timer = setTimeout(finish, milliseconds);
    onlineEvents.addEventListener?.("online", finish, { once: true });
  });
}

async function responseError(response: Response): Promise<Error> {
  const body = await response.text();
  try {
    const parsed = JSON.parse(body) as { error?: { code?: string; message?: string } };
    return new ChatStreamError(
      parsed.error?.code ?? `http_${response.status}`,
      parsed.error?.message ?? `HTTP ${response.status}`,
    );
  } catch {
    return new ChatStreamError(`http_${response.status}`, body.trim() || `HTTP ${response.status}`);
  }
}
