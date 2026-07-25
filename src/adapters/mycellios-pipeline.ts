import {
  AdapterError,
  type AdapterCapabilities,
  type AdapterChunk,
  type AdapterMetrics,
  type AdapterRequest,
  type InferenceAdapter,
} from "./base.js";

export interface MycelliosPipelineOptions {
  baseUrl: string;
  model: string;
  modelDigest: string;
  activationId: string;
}

/**
 * Local gateway for a pipeline launched and authenticated by Mycellios.
 *
 * This is deliberately not a generic OpenAI-compatible adapter. It accepts
 * only a credential-free loopback origin, validates the native runtime health
 * identity before advertising the model, and exposes no host allowlist or API
 * key escape hatch.
 */
export class MycelliosPipelineAdapter implements InferenceAdapter {
  readonly kind = "mycellios-pipeline" as const;
  private readonly baseUrl: URL;
  private activeJobs = 0;
  private ready = false;

  constructor(private readonly options: MycelliosPipelineOptions) {
    this.baseUrl = normalizeMycelliosPipelineBaseUrl(options.baseUrl);
  }

  async probe(): Promise<AdapterCapabilities> {
    const response = await fetch(new URL("health", this.baseUrl), {
      signal: AbortSignal.timeout(5_000),
      redirect: "manual",
    });
    if (!response.ok) {
      this.ready = false;
      throw new AdapterError(
        `Mycellios pipeline health probe returned HTTP ${response.status}`,
        "mycellios_pipeline_probe_failed",
        true,
      );
    }
    const health = await parseJsonRecord(response, "mycellios_pipeline_health_invalid");
    if (
      health.status !== "ready"
      || health.model !== this.options.model
      || health.artifact_identity !== this.options.modelDigest
      || health.pipeline_snapshot_identity !== this.options.activationId
      || !Number.isInteger(health.stages)
      || (health.stages as number) < 1
      || !Array.isArray(health.boundaries)
    ) {
      this.ready = false;
      throw new AdapterError(
        "The loopback endpoint is not the expected native Mycellios pipeline",
        "mycellios_pipeline_identity_mismatch",
        false,
      );
    }
    this.ready = true;
    return {
      kind: this.kind,
      models: [this.options.model],
      streaming: true,
      embeddings: false,
      reranking: false,
    };
  }

  async warm(model: string): Promise<void> {
    if (model !== this.options.model) {
      throw new AdapterError(
        `Mycellios pipeline does not own model ${model}`,
        "mycellios_pipeline_model_mismatch",
        false,
      );
    }
    await this.probe();
  }

  async *generate(request: AdapterRequest, signal: AbortSignal): AsyncIterable<AdapterChunk> {
    if (!this.ready) {
      throw new AdapterError(
        "Mycellios pipeline identity has not been verified",
        "mycellios_pipeline_not_verified",
        true,
      );
    }
    if (request.request.model !== this.options.model) {
      throw new AdapterError(
        `Mycellios pipeline does not own model ${request.request.model}`,
        "mycellios_pipeline_model_mismatch",
        false,
      );
    }
    this.activeJobs += 1;
    try {
      const sessionId = request.request.session_id;
      const response = await fetch(new URL("v1/chat/completions", this.baseUrl), {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(sessionId ? { "x-session-id": sessionId } : {}),
        },
        body: JSON.stringify({
          model: this.options.model,
          messages: request.request.messages,
          max_tokens: request.request.max_tokens,
          temperature: request.request.temperature,
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
          `Mycellios pipeline generation failed (${response.status}): ${detail.slice(0, 300)}`,
          "mycellios_pipeline_generation_failed",
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

  async cancel(): Promise<void> {
    // Cancellation is propagated by aborting the request signal.
  }

  async metrics(): Promise<AdapterMetrics> {
    return {
      ready: this.ready,
      activeJobs: this.activeJobs,
      loadedModels: this.ready ? [this.options.model] : [],
    };
  }
}

export function normalizeMycelliosPipelineBaseUrl(rawUrl: string): URL {
  const normalized = rawUrl.endsWith("/") ? rawUrl : `${rawUrl}/`;
  const url = new URL(normalized);
  const loopback = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
  if (
    !new Set(["http:", "https:"]).has(url.protocol)
    || !loopback.has(url.hostname.toLowerCase())
    || url.username !== ""
    || url.password !== ""
    || url.search !== ""
    || url.hash !== ""
    || url.pathname !== "/"
  ) {
    throw new AdapterError(
      "Mycellios pipeline endpoints must be credential-free loopback origins",
      "mycellios_pipeline_endpoint_not_local",
      false,
    );
  }
  return url;
}

async function parseJsonRecord(
  response: Response,
  errorCode: string,
): Promise<Record<string, unknown>> {
  try {
    const value = await response.json() as unknown;
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error();
    return value as Record<string, unknown>;
  } catch {
    throw new AdapterError(
      "Mycellios pipeline returned invalid identity evidence",
      errorCode,
      false,
    );
  }
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
        assertStreamBufferSize(event);
        const data = event
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data) yield data;
        boundary = buffer.indexOf("\n\n");
      }
      assertStreamBufferSize(buffer);
    }
  } finally {
    reader.releaseLock();
  }
}

function assertStreamBufferSize(value: string): void {
  if (Buffer.byteLength(value, "utf8") > 1024 * 1024) {
    throw new AdapterError(
      "Mycellios pipeline stream buffer exceeded 1 MiB",
      "stream_buffer_limit",
      false,
    );
  }
}
