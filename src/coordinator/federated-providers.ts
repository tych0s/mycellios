import type {
  FederatedInferenceEvent,
  FederatedInferenceInput,
  FederatedNetworkClass,
  FederatedNetworkId,
  FederatedProviderAdapter,
  FederatedProviderModel,
  FederatedProviderNode,
} from "../contracts/federation.js";
import type { ChatCompletionRequest } from "../contracts/types.js";
import type { FederationRuntimeConfig } from "../core/config.js";
import { spawn, type ChildProcess } from "node:child_process";

interface OpenAiProviderOptions {
  id: FederatedNetworkId;
  class: FederatedNetworkClass;
  baseUrl: string;
  apiKey?: string | undefined;
  requiresKey: boolean;
  defaultInputUsdPerMillion?: number;
  defaultOutputUsdPerMillion?: number;
  experimental?: boolean;
}

export function createFederatedProviderAdapters(
  config: FederationRuntimeConfig,
): FederatedProviderAdapter[] {
  return [
    new ExternalRuntimeAAdapter(
      config.external-runtime-aInferenceUrl,
      config.external-runtime-aManagementUrl,
      config.external-runtime-aExecutable,
    ),
    new AiHordeAdapter(config.aiHordeBaseUrl, config.aiHordeApiKey),
    new OpenAiFederatedAdapter({
      id: "peer-runtime",
      class: "community",
      baseUrl: config.peer-runtimeBaseUrl,
      requiresKey: false,
      experimental: true,
    }),
    new OpenAiFederatedAdapter({
      id: "chutes",
      class: "token-api",
      baseUrl: config.chutesBaseUrl,
      apiKey: config.chutesApiKey,
      requiresKey: true,
      defaultInputUsdPerMillion: 1,
      defaultOutputUsdPerMillion: 2,
    }),
    new OpenAiFederatedAdapter({
      id: "akashml",
      class: "token-api",
      baseUrl: config.akashMlBaseUrl,
      apiKey: config.akashMlApiKey,
      requiresKey: true,
      defaultInputUsdPerMillion: 1,
      defaultOutputUsdPerMillion: 2,
    }),
  ];
}

class OpenAiFederatedAdapter implements FederatedProviderAdapter {
  readonly id: FederatedNetworkId;
  readonly class: FederatedNetworkClass;
  readonly configured: boolean;
  private readonly baseUrl: string;
  private readonly apiKey: string | undefined;
  private readonly defaultInputUsdPerMillion: number;
  private readonly defaultOutputUsdPerMillion: number;

  constructor(options: OpenAiProviderOptions) {
    this.id = options.id;
    this.class = options.class;
    this.baseUrl = options.baseUrl;
    this.apiKey = options.apiKey;
    this.configured = !options.requiresKey || Boolean(options.apiKey);
    this.defaultInputUsdPerMillion = options.defaultInputUsdPerMillion ?? 0;
    this.defaultOutputUsdPerMillion = options.defaultOutputUsdPerMillion ?? 0;
  }

  async discover(signal: AbortSignal): Promise<{
    models: FederatedProviderModel[];
    nodes: FederatedProviderNode[];
  }> {
    this.assertConfigured();
    const response = await fetch(`${this.baseUrl}/models`, {
      headers: this.headers(),
      signal,
    });
    if (!response.ok) throw await providerHttpError(this.id, response);
    const body = await response.json() as {
      data?: Array<{
        id?: unknown;
        name?: unknown;
        context_length?: unknown;
        context_length_tokens?: unknown;
        input_price?: unknown;
        output_price?: unknown;
      }>;
    };
    const models = (body.data ?? []).flatMap((model) => {
      if (typeof model.id !== "string" || !model.id.trim()) return [];
      return [{
        canonicalId: model.id,
        externalId: model.id,
        displayName: typeof model.name === "string" ? model.name : model.id,
        ...finitePositive(model.context_length ?? model.context_length_tokens, "contextTokens"),
        estimatedInputUsdPerMillion: pricePerMillion(
          model.input_price,
          this.defaultInputUsdPerMillion,
        ),
        estimatedOutputUsdPerMillion: pricePerMillion(
          model.output_price,
          this.defaultOutputUsdPerMillion,
        ),
      }];
    });
    return {
      models,
      nodes: [{
        externalId: `${this.id}-aggregate`,
        label: `${providerName(this.id)} capacity`,
        scope: "aggregate",
        models: models.map((model) => model.canonicalId),
        status: models.length > 0 ? "online" : "unknown",
        individuallySelectable: false,
      }],
    };
  }

  async *infer(input: FederatedInferenceInput): AsyncIterable<FederatedInferenceEvent> {
    this.assertConfigured();
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        ...this.headers(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: input.externalModel,
        messages: input.request.messages.map(({ role, content }) => ({
          role,
          content,
        })),
        stream: true,
        max_tokens: input.request.max_tokens,
        temperature: input.request.temperature,
        top_p: input.request.top_p,
        seed: input.request.seed,
      }),
      signal: input.signal,
    });
    if (!response.ok) throw await providerHttpError(this.id, response);
    if (!response.body) throw new Error(`${this.id}_empty_stream`);
    let text = "";
    let index = 0;
    let finishReason = "stop";
    let usageInputTokens: number | undefined;
    let usageOutputTokens: number | undefined;
    let reportedCostUsd: number | undefined;
    for await (const data of readSseData(response.body)) {
      if (data === "[DONE]") break;
      const parsed = JSON.parse(data) as {
        choices?: Array<{
          delta?: { content?: unknown };
          message?: { content?: unknown };
          finish_reason?: unknown;
        }>;
        usage?: {
          prompt_tokens?: unknown;
          completion_tokens?: unknown;
          cost?: unknown;
        };
      };
      const choice = parsed.choices?.[0];
      const fragment = typeof choice?.delta?.content === "string"
        ? choice.delta.content
        : typeof choice?.message?.content === "string"
          ? choice.message.content
          : "";
      if (fragment) {
        text += fragment;
        yield { type: "token", text: fragment, index, at: Date.now() };
        index += 1;
      }
      if (typeof choice?.finish_reason === "string") finishReason = choice.finish_reason;
      usageInputTokens = finiteNumber(parsed.usage?.prompt_tokens) ?? usageInputTokens;
      usageOutputTokens = finiteNumber(parsed.usage?.completion_tokens) ?? usageOutputTokens;
      reportedCostUsd = finiteNumber(parsed.usage?.cost) ?? reportedCostUsd;
    }
    yield {
      type: "completed",
      text,
      inputTokens: usageInputTokens ?? estimateInputTokens(input.request),
      outputTokens: usageOutputTokens ?? estimateTextTokens(text),
      finishReason,
      ...(reportedCostUsd === undefined ? {} : { actualCostUsd: reportedCostUsd }),
    };
  }

  estimateMaximumCostUsd(input: {
    request: ChatCompletionRequest;
    model: FederatedProviderModel;
  }): number {
    const inputTokens = estimateInputTokens(input.request);
    const outputTokens = input.request.max_tokens ?? 512;
    return (
      inputTokens * (input.model.estimatedInputUsdPerMillion ?? this.defaultInputUsdPerMillion)
      + outputTokens * (input.model.estimatedOutputUsdPerMillion ?? this.defaultOutputUsdPerMillion)
    ) / 1_000_000;
  }

  private assertConfigured(): void {
    if (!this.configured) throw new Error(`${this.id}_not_configured`);
  }

  private headers(): Record<string, string> {
    return this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {};
  }
}

class ExternalRuntimeAAdapter implements FederatedProviderAdapter {
  readonly id = "external-runtime-a" as const;
  readonly class = "community" as const;
  readonly configured = true;

  private child: ChildProcess | null = null;
  private starting: Promise<void> | null = null;

  constructor(
    private readonly inferenceUrl: string,
    private readonly managementUrl: string,
    private readonly executable: string | undefined,
  ) {}

  async discover(signal: AbortSignal): Promise<{
    models: FederatedProviderModel[];
    nodes: FederatedProviderNode[];
  }> {
    await this.ensureClient(signal);
    const modelsRequest = this.executable
      ? this.fetchModelsWhenReady(signal)
      : fetch(`${this.inferenceUrl}/v1/models`, { signal });
    const [modelsResponse, status] = await Promise.all([
      modelsRequest,
      fetch(`${this.managementUrl}/api/status`, { signal })
        .then(async (response) =>
          response.ok
            ? await response.json() as Record<string, unknown>
            : {})
        .catch((error: unknown) => {
          if (signal.aborted) throw error;
          // Some external runtime A client builds expose the inference gateway without
          // opening the optional management console. Models remain advertised
          // only and still need a real generation before becoming routable.
          return {};
        }),
    ]);
    if (!modelsResponse.ok) throw await providerHttpError(this.id, modelsResponse);
    const modelBody = await modelsResponse.json() as {
      data?: Array<{ id?: unknown }>;
    };
    const models = (modelBody.data ?? []).flatMap((entry) =>
      typeof entry.id === "string" && entry.id.trim()
        ? [{ canonicalId: entry.id, externalId: entry.id, displayName: entry.id }]
        : []);
    const peerEntries = findPeerEntries(status);
    const nodes = peerEntries.length > 0
      ? peerEntries.map((peer, index): FederatedProviderNode => ({
          externalId: String(peer.id ?? peer.peer_id ?? `peer-${index}`),
          label: `Mesh peer ${index + 1}`,
          scope: "logical",
          models: models.map((model) => model.canonicalId),
          status: "online",
          ...optionalNumber("reliability", finiteNumber(peer.reliability)),
          individuallySelectable: false,
        }))
      : [{
          externalId: "external-runtime-a-aggregate",
          label: "external runtime A capacity",
          scope: "aggregate" as const,
          models: models.map((model) => model.canonicalId),
          status: models.length > 0 ? "online" as const : "unknown" as const,
          individuallySelectable: false,
        }];
    return { models, nodes };
  }

  infer(input: FederatedInferenceInput): AsyncIterable<FederatedInferenceEvent> {
    return new OpenAiFederatedAdapter({
      id: this.id,
      class: this.class,
      baseUrl: `${this.inferenceUrl}/v1`,
      requiresKey: false,
    }).infer(input);
  }

  estimateMaximumCostUsd(): number {
    return 0;
  }

  async close(): Promise<void> {
    const child = this.child;
    this.child = null;
    this.starting = null;
    if (!child || child.exitCode !== null) return;
    child.kill("SIGTERM");
    await Promise.race([
      new Promise<void>((resolveExit) => child.once("exit", () => resolveExit())),
      new Promise<void>((resolveDelay) => setTimeout(resolveDelay, 5_000)),
    ]);
    if (child.exitCode === null) child.kill("SIGKILL");
  }

  private ensureClient(signal: AbortSignal): Promise<void> {
    if (!this.executable || this.child?.exitCode === null) return Promise.resolve();
    if (this.starting) return this.starting;
    this.starting = new Promise<void>((resolveStart, rejectStart) => {
      const child = spawn(this.executable!, ["client", "--auto"], {
        shell: false,
        windowsHide: true,
        stdio: "ignore",
        env: minimalSidecarEnvironment(process.env),
      });
      this.child = child;
      const onAbort = () => {
        child.kill("SIGTERM");
        rejectStart(signal.reason ?? new Error("external-runtime-a_start_aborted"));
      };
      signal.addEventListener("abort", onAbort, { once: true });
      child.once("error", (error) => {
        signal.removeEventListener("abort", onAbort);
        this.child = null;
        rejectStart(new Error(`external-runtime-a_client_start_failed:${error.message}`));
      });
      // Successful spawn is enough here. The bounded discovery requests below
      // are the readiness check and will keep advertised capacity unroutable
      // until both APIs and a real generation succeed.
      child.once("spawn", () => {
        signal.removeEventListener("abort", onAbort);
        resolveStart();
      });
      child.once("exit", () => {
        if (this.child === child) this.child = null;
      });
    }).finally(() => {
      this.starting = null;
    });
    return this.starting;
  }

  private async fetchModelsWhenReady(signal: AbortSignal): Promise<Response> {
    let lastError: unknown = new Error("external-runtime-a_gateway_not_ready");
    for (let attempt = 0; attempt < 20; attempt += 1) {
      try {
        const response = await fetch(`${this.inferenceUrl}/v1/models`, { signal });
        if (response.ok || response.status < 500 && response.status !== 404) return response;
        lastError = new Error(`external-runtime-a_gateway_http_${response.status}`);
      } catch (error) {
        if (signal.aborted) throw error;
        lastError = error;
      }
      await abortableDelay(500, signal);
    }
    throw lastError;
  }
}

class AiHordeAdapter implements FederatedProviderAdapter {
  readonly id = "ai-horde" as const;
  readonly class = "community" as const;
  readonly configured = true;

  constructor(
    private readonly baseUrl: string,
    private readonly apiKey?: string,
  ) {}

  async discover(signal: AbortSignal): Promise<{
    models: FederatedProviderModel[];
    nodes: FederatedProviderNode[];
  }> {
    const headers = this.headers();
    const [modelsResponse, workersResponse] = await Promise.all([
      fetch(`${this.baseUrl}/v2/status/models?type=text`, { headers, signal }),
      fetch(`${this.baseUrl}/v2/workers?type=text`, { headers, signal }),
    ]);
    if (!modelsResponse.ok) throw await providerHttpError(this.id, modelsResponse);
    if (!workersResponse.ok) throw await providerHttpError(this.id, workersResponse);
    const rawModels = await modelsResponse.json() as Array<Record<string, unknown>>;
    const rawWorkers = await workersResponse.json() as Array<Record<string, unknown>>;
    const models = [...rawModels]
      .sort((left, right) =>
        (finiteNumber(left.eta) ?? Number.MAX_SAFE_INTEGER)
          - (finiteNumber(right.eta) ?? Number.MAX_SAFE_INTEGER)
        || (finiteNumber(left.queued) ?? Number.MAX_SAFE_INTEGER)
          - (finiteNumber(right.queued) ?? Number.MAX_SAFE_INTEGER)
        || (finiteNumber(right.count) ?? 0) - (finiteNumber(left.count) ?? 0))
      .flatMap((entry) => {
        const name = typeof entry.name === "string" ? entry.name : null;
        return name ? [{ canonicalId: name, externalId: name, displayName: name }] : [];
      });
    const nodes = rawWorkers.map((worker, index): FederatedProviderNode => ({
      externalId: String(worker.id ?? `worker-${index}`),
      label: `AI Horde worker ${index + 1}`,
      scope: "logical",
      models: Array.isArray(worker.models)
        ? worker.models.filter((model): model is string => typeof model === "string")
        : [],
      status: worker.maintenance_mode === true ? "degraded" : "online",
      ...optionalNumber("reliability", finiteNumber(worker.reliability)),
      individuallySelectable: false,
    }));
    return { models, nodes };
  }

  async *infer(input: FederatedInferenceInput): AsyncIterable<FederatedInferenceEvent> {
    const response = await fetch(`${this.baseUrl}/v2/generate/text/async`, {
      method: "POST",
      headers: { ...this.headers(), "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: requestPrompt(input.request),
        models: [input.externalModel],
        params: {
          max_length: input.request.max_tokens ?? 512,
          temperature: input.request.temperature,
          top_p: input.request.top_p,
          seed: input.request.seed,
        },
      }),
      signal: input.signal,
    });
    if (!response.ok) throw await providerHttpError(this.id, response);
    const accepted = await response.json() as { id?: unknown };
    if (typeof accepted.id !== "string") throw new Error("ai_horde_missing_request_id");
    const generationId = accepted.id;
    try {
      while (true) {
        await abortableDelay(1_000, input.signal);
        const statusResponse = await fetch(
          `${this.baseUrl}/v2/generate/text/status/${encodeURIComponent(generationId)}`,
          { headers: this.headers(), signal: input.signal },
        );
        if (!statusResponse.ok) throw await providerHttpError(this.id, statusResponse);
        const status = await statusResponse.json() as {
          done?: boolean;
          faulted?: boolean;
          generations?: Array<{ text?: unknown }>;
        };
        if (status.faulted) throw new Error("ai_horde_generation_faulted");
        if (!status.done) {
          yield { type: "heartbeat", at: Date.now() };
          continue;
        }
        const text = typeof status.generations?.[0]?.text === "string"
          ? status.generations[0].text
          : "";
        // AI Horde text is asynchronous. Emit one final buffered fragment while
        // heartbeats above keep the SSE connection alive.
        if (text) yield { type: "token", text, index: 0, at: Date.now() };
        yield {
          type: "completed",
          text,
          inputTokens: estimateInputTokens(input.request),
          outputTokens: estimateTextTokens(text),
          finishReason: "stop",
        };
        return;
      }
    } catch (error) {
      if (input.signal.aborted) {
        await fetch(
          `${this.baseUrl}/v2/generate/text/status/${encodeURIComponent(generationId)}`,
          { method: "DELETE", headers: this.headers() },
        ).catch(() => undefined);
      }
      throw error;
    }
  }

  estimateMaximumCostUsd(): number {
    return 0;
  }

  private headers(): Record<string, string> {
    return {
      apikey: this.apiKey || "0000000000",
      "Client-Agent": "mycellios:1:contact@mycellios.com",
    };
  }
}

async function* readSseData(
  stream: ReadableStream<Uint8Array>,
): AsyncIterable<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      while (true) {
        const boundary = buffer.indexOf("\n\n");
        if (boundary < 0) break;
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = frame.split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data) yield data;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function requestPrompt(request: ChatCompletionRequest): string {
  return request.messages.map((message) => `${message.role}: ${message.content}`).join("\n");
}

function estimateInputTokens(request: ChatCompletionRequest): number {
  return estimateTextTokens(requestPrompt(request));
}

function estimateTextTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

function pricePerMillion(value: unknown, fallback: number): number {
  const number = finiteNumber(value);
  if (number === undefined) return fallback;
  // Providers commonly expose price per token. Values above one are already
  // treated as per-million pricing.
  return number > 1 ? number : number * 1_000_000;
}

function finiteNumber(value: unknown): number | undefined {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function finitePositive(
  value: unknown,
  key: "contextTokens",
): Partial<Record<typeof key, number>> {
  const number = finiteNumber(value);
  return number !== undefined && number > 0 ? { [key]: number } : {};
}

function optionalNumber<K extends string>(
  key: K,
  value: number | undefined,
): Partial<Record<K, number>> {
  return value === undefined ? {} : { [key]: value } as Record<K, number>;
}

function findPeerEntries(status: Record<string, unknown>): Array<Record<string, unknown>> {
  for (const key of ["peers", "nodes", "connections"]) {
    const value = status[key];
    if (Array.isArray(value)) {
      return value.filter(
        (entry): entry is Record<string, unknown> =>
          typeof entry === "object" && entry !== null,
      );
    }
  }
  return [];
}

function providerName(id: FederatedNetworkId): string {
  return ({
    external-runtime-a: "external runtime A",
    "ai-horde": "AI Horde",
    peer-runtime: "peer runtime",
    chutes: "Chutes",
    akashml: "AkashML",
    gpu_cloud: "GpuCloudCloud",
    vast: "Vast.ai",
    clore: "Clore.ai",
  })[id];
}

async function providerHttpError(
  provider: FederatedNetworkId,
  response: Response,
): Promise<Error> {
  await response.body?.cancel().catch(() => undefined);
  return new Error(`${provider}_http_${response.status}`);
}

function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason ?? new Error("aborted"));
      return;
    }
    const timer = setTimeout(resolve, milliseconds);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason ?? new Error("aborted"));
    }, { once: true });
  });
}

function minimalSidecarEnvironment(
  environment: NodeJS.ProcessEnv,
): NodeJS.ProcessEnv {
  const allowed = [
    "PATH",
    "Path",
    "PATHEXT",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "HOME",
    "USERPROFILE",
    "TMP",
    "TEMP",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
  ] as const;
  return Object.fromEntries(
    allowed.flatMap((key) => environment[key] === undefined
      ? []
      : [[key, environment[key]]]),
  );
}
