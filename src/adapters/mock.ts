import type {
  AdapterCapabilities,
  AdapterChunk,
  AdapterMetrics,
  AdapterRequest,
  InferenceAdapter,
} from "./base.js";

export interface MockAdapterOptions {
  model: string;
  tokensPerSecond: number;
  ttftMs: number;
  failureRate: number;
  seed?: number;
}

export class MockAdapter implements InferenceAdapter {
  readonly kind = "mock" as const;
  private activeJobs = 0;

  constructor(private readonly options: MockAdapterOptions) {}

  async probe(): Promise<AdapterCapabilities> {
    return {
      kind: this.kind,
      models: [this.options.model],
      streaming: true,
      embeddings: false,
      reranking: false,
    };
  }

  async warm(_model: string): Promise<void> {}

  async *generate(request: AdapterRequest, signal: AbortSignal): AsyncIterable<AdapterChunk> {
    this.activeJobs += 1;
    try {
      await abortableDelay(this.options.ttftMs, signal);
      const random = mulberry32((request.request.seed ?? this.options.seed ?? 42) >>> 0);
      if (random() < this.options.failureRate) {
        throw new Error("Synthetic worker failure requested by mock profile");
      }
      const lastUser = [...request.request.messages]
        .reverse()
        .find((message) => message.role === "user")?.content;
      const response =
        `Response from ${this.options.model}, executed by the distributed network. ` +
        `I received: ${lastUser ?? "a request without a user message"}. ` +
        "This result comes from the deterministic test adapter.";
      const pieces = response.match(/\S+\s*/g) ?? [response];
      const maxPieces = Math.min(pieces.length, request.request.max_tokens ?? pieces.length);
      const delayMs = Math.max(0, Math.round(1_000 / this.options.tokensPerSecond));
      for (let index = 0; index < maxPieces; index += 1) {
        if (signal.aborted) throw signal.reason ?? new Error("Generation cancelled");
        if (delayMs > 0) await abortableDelay(delayMs, signal);
        yield { index, text: pieces[index]! };
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
}

function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (milliseconds <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, milliseconds);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason ?? new Error("Operation aborted"));
      },
      { once: true },
    );
  });
}

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let result = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    result = (result + Math.imul(result ^ (result >>> 7), 61 | result)) ^ result;
    return ((result ^ (result >>> 14)) >>> 0) / 4_294_967_296;
  };
}
