import type { ChatResponse, ChatStreamUpdate } from "./contracts.js";

interface OpenAiStreamChunk {
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
  "first_token_timeout",
  "pipeline_stage_disconnected",
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
}

export async function consumeChatCompletionStreamWithRecovery(
  openResponse: (attempt: number) => Promise<Response>,
  fallbackModel: string,
  onUpdate: (update: ChatStreamUpdate) => void,
  options: RecoveringChatStreamOptions,
): Promise<ChatResponse> {
  const maximumAttempts = Math.max(1, Math.min(2, options.maximumAttempts ?? 2));
  const retryDelayMs = Math.max(0, options.retryDelayMs ?? 2_500);
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
  };
  onUpdate(latest);
  for (let attempt = 1; attempt <= maximumAttempts; attempt += 1) {
    const startedAt = Date.now();
    try {
      const response = await openResponse(attempt);
      return await consumeChatCompletionStream(
        response,
        fallbackModel,
        (update) => {
          latest = {
            ...update,
            ...(update.outputTokens > 0
              ? { phase: "streaming" as const }
              : update.phase
                ? { phase: update.phase }
                : {}),
            attempt,
          };
          onUpdate(latest);
        },
        startedAt,
      );
    } catch (error) {
      if (
        attempt >= maximumAttempts ||
        latest.outputTokens > 0 ||
        !isRecoverablePretokenStreamError(error)
      ) {
        throw error;
      }
      latest = {
        ...latest,
        requestId: "pending",
        phase: "recovering",
        statusMessage: "A route node disconnected before the first token. Reconnecting and retrying once.",
        attempt: attempt + 1,
        elapsedMs: Math.max(latest.elapsedMs, Date.now() - startedAt),
      };
      onUpdate(latest);
      await delay(retryDelayMs);
    }
  }
  throw new ChatStreamError("retry_exhausted", "The inference retry was exhausted.");
}

export async function consumeChatCompletionStream(
  response: Response,
  fallbackModel: string,
  onUpdate: (update: ChatStreamUpdate) => void,
  startedAt = Date.now(),
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

    let chunk: OpenAiStreamChunk;
    try {
      chunk = JSON.parse(data) as OpenAiStreamChunk;
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
    const { done, value } = await reader.read();
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

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
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
