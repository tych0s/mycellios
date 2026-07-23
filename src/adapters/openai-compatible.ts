import { AdapterError } from "./base.js";
import type {
  AdapterCapabilities,
  AdapterChunk,
  AdapterMetrics,
  AdapterRequest,
  InferenceAdapter,
} from "./base.js";

export interface OpenAICompatibleOptions {
  kind?: "externalggufruntime" | "openai-compatible";
  baseUrl: string;
  model: string;
  apiPathPrefix?: string;
  apiKey?: string;
  allowedHosts?: string[];
  requestTemperature?: number | undefined;
}

export class OpenAICompatibleAdapter implements InferenceAdapter {
  readonly kind: "externalggufruntime" | "openai-compatible";
  private activeJobs = 0;
  private readonly baseUrl: URL;

  constructor(private readonly options: OpenAICompatibleOptions) {
    this.kind = options.kind ?? "openai-compatible";
    this.baseUrl = normalizeAndValidateBaseUrl(options.baseUrl, options.allowedHosts ?? []);
    if (options.apiKey && this.baseUrl.protocol !== "https:") {
      throw new AdapterError(
        "An OpenAI-compatible backend with an API key must use HTTPS",
        "insecure_backend_credentials",
        false,
      );
    }
  }

  async probe(): Promise<AdapterCapabilities> {
    const response = await fetch(this.endpoint("models"), {
      headers: this.headers(),
      signal: AbortSignal.timeout(5_000),
      redirect: "manual",
    });
    if (!response.ok) {
      throw new AdapterError(
        `Backend model probe returned HTTP ${response.status}`,
        "backend_probe_failed",
        true,
      );
    }
    return {
      kind: this.kind,
      models: [this.options.model],
      streaming: true,
      embeddings: this.kind === "externalggufruntime",
      reranking: this.kind === "externalggufruntime",
    };
  }

  async warm(_model: string): Promise<void> {
    await this.probe();
  }

  async *generate(request: AdapterRequest, signal: AbortSignal): AsyncIterable<AdapterChunk> {
    this.activeJobs += 1;
    try {
      const sessionId = request.request.session_id;
      const response = await fetch(this.endpoint("chat/completions"), {
        method: "POST",
        headers: {
          ...this.headers(),
          "content-type": "application/json",
          ...(sessionId ? { "x-session-id": sessionId } : {}),
        },
        body: JSON.stringify({
          model: this.options.model,
          messages: request.request.messages,
          max_tokens: request.request.max_tokens,
          temperature: this.options.requestTemperature ?? request.request.temperature,
          top_p: request.request.top_p,
          seed: request.request.seed,
          stream: true,
        }),
        signal,
        redirect: "manual",
      });
      if (!response.ok || !response.body) {
        const detail = await response.text();
        throw new AdapterError(
          `Backend generation failed (${response.status}): ${detail.slice(0, 300)}`,
          "backend_generation_failed",
          response.status >= 500,
        );
      }
      let index = 0;
      for await (const payload of readSseData(response.body)) {
        if (payload === "[DONE]") break;
        const parsed = JSON.parse(payload) as {
          choices?: Array<{ delta?: { content?: string }; text?: string }>;
          usage?: {
            prompt_tokens?: number;
            completion_tokens?: number;
          };
          distribution_metrics?: {
            ttft_ms?: number;
            pipeline_ms?: number;
            reused_kv_tokens?: number;
          };
        };
        const text = parsed.choices?.[0]?.delta?.content ?? parsed.choices?.[0]?.text ?? "";
        if (text) yield { index: index++, text };
        const metrics = parsed.distribution_metrics;
        if (metrics || parsed.usage) {
          yield {
            index,
            text: "",
            metrics: {
              inputTokens: parsed.usage?.prompt_tokens,
              outputTokens: parsed.usage?.completion_tokens,
              ttftMs: metrics?.ttft_ms,
              activeMs: metrics?.pipeline_ms,
              reusedKvTokens: metrics?.reused_kv_tokens,
            },
          };
        }
      }
    } finally {
      this.activeJobs -= 1;
    }
  }

  async cancel(_jobId: string): Promise<void> {}

  async metrics(): Promise<AdapterMetrics> {
    return {
      ready: true,
      activeJobs: this.activeJobs,
      loadedModels: [this.options.model],
    };
  }

  private headers(): Record<string, string> {
    return this.options.apiKey ? { authorization: `Bearer ${this.options.apiKey}` } : {};
  }

  private endpoint(path: string): URL {
    const prefix = (this.options.apiPathPrefix ?? "v1").replace(/^\/+|\/+$/g, "");
    return new URL(`${prefix ? `${prefix}/` : ""}${path}`, this.baseUrl);
  }
}

export function normalizeAndValidateBaseUrl(rawUrl: string, allowedHosts: string[]): URL {
  const normalized = rawUrl.endsWith("/") ? rawUrl : `${rawUrl}/`;
  const url = new URL(normalized);
  if (!new Set(["http:", "https:"]).has(url.protocol)) {
    throw new AdapterError("Only HTTP(S) inference backends are allowed", "invalid_backend_url", false);
  }
  const host = url.hostname.toLowerCase();
  const loopback = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
  const allow = new Set(allowedHosts.map((entry) => entry.toLowerCase()));
  if (!loopback.has(host) && !allow.has(host)) {
    throw new AdapterError(
      `Backend host ${host} is not in the explicit allowlist`,
      "backend_host_not_allowed",
      false,
    );
  }
  if (!loopback.has(host) && url.protocol !== "https:") {
    throw new AdapterError(
      "Remote inference backends must use HTTPS",
      "insecure_remote_backend",
      false,
    );
  }
  return url;
}

async function* readSseData(stream: ReadableStream<Uint8Array>): AsyncIterable<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replaceAll("\r\n", "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const event = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        assertStreamBufferSize(event, "SSE event");
        const data = event
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data) yield data;
        boundary = buffer.indexOf("\n\n");
      }
      assertStreamBufferSize(buffer, "SSE");
    }
  } finally {
    reader.releaseLock();
  }
}

function assertStreamBufferSize(value: string, label: string): void {
  if (Buffer.byteLength(value, "utf8") > 1024 * 1024) {
    throw new AdapterError(
      `${label} buffer exceeded 1 MiB`,
      "stream_buffer_limit",
      false,
    );
  }
}
