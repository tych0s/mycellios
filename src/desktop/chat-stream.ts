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
  };
  error?: { message?: string };
}

export async function consumeChatCompletionStream(
  response: Response,
  fallbackModel: string,
  onUpdate: (update: ChatStreamUpdate) => void,
  startedAt = Date.now(),
): Promise<ChatResponse> {
  if (!response.ok) throw new Error(await responseError(response));
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

  const emit = (delta: string) => onUpdate({
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
    if (chunk.error?.message) throw new Error(chunk.error.message);
    if (chunk.id) requestId = chunk.id;
    if (chunk.model) model = chunk.model;
    const receivedRoute = chunk.x_network?.route_class !== undefined;
    if (chunk.x_network?.route_class) routeClass = chunk.x_network.route_class;
    if (chunk.x_network?.affinity_hit !== undefined) affinityHit = chunk.x_network.affinity_hit;
    if (chunk.x_network?.session_id) sessionId = chunk.x_network.session_id;
    if (chunk.x_network?.reused_kv_tokens !== undefined) {
      reusedKvTokens = chunk.x_network.reused_kv_tokens;
    }

    const delta = chunk.choices?.[0]?.delta?.content ?? "";
    if (delta) {
      text += delta;
      outputTokens = chunk.x_network?.token_index === undefined
        ? outputTokens + 1
        : Math.max(outputTokens, chunk.x_network.token_index + 1);
      if (ttftMs === 0) ttftMs = Math.max(1, Date.now() - startedAt);
      emit(delta);
    }

    if (chunk.usage) {
      promptTokens = chunk.usage.prompt_tokens ?? promptTokens;
      outputTokens = chunk.usage.completion_tokens ?? outputTokens;
      totalTokens = chunk.usage.total_tokens ?? promptTokens + outputTokens;
      ttftMs = chunk.x_network?.ttft_ms ?? ttftMs;
      activeMs = chunk.x_network?.active_ms ?? Math.max(0, Date.now() - startedAt);
      completed = true;
      emit("");
    } else if (!delta && receivedRoute) {
      emit("");
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

async function responseError(response: Response): Promise<string> {
  const body = await response.text();
  try {
    const parsed = JSON.parse(body) as { error?: { message?: string } };
    return parsed.error?.message ?? `HTTP ${response.status}`;
  } catch {
    return body.trim() || `HTTP ${response.status}`;
  }
}
