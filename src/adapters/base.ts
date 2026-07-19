import type { AdapterKind, ChatCompletionRequest } from "../contracts/types.js";

export interface AdapterCapabilities {
  kind: AdapterKind;
  models: string[];
  streaming: boolean;
  embeddings: boolean;
  reranking: boolean;
}

export interface AdapterMetrics {
  ready: boolean;
  activeJobs: number;
  loadedModels: string[];
}

export interface AdapterChunk {
  index: number;
  text: string;
}

export interface AdapterRequest {
  jobId: string;
  request: ChatCompletionRequest;
}

export interface InferenceAdapter {
  readonly kind: AdapterKind;
  probe(): Promise<AdapterCapabilities>;
  warm(model: string): Promise<void>;
  generate(request: AdapterRequest, signal: AbortSignal): AsyncIterable<AdapterChunk>;
  cancel(jobId: string): Promise<void>;
  metrics(): Promise<AdapterMetrics>;
}

export class AdapterError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly retryable: boolean,
  ) {
    super(message);
    this.name = "AdapterError";
  }
}
