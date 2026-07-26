import {
  AdapterError,
  type AdapterCapabilities,
  type AdapterChunk,
  type AdapterMetrics,
  type AdapterRequest,
  type InferenceAdapter,
} from "./base.js";

/**
 * Control-plane adapter for a native Mycellios contributor.
 *
 * The actual model stages are launched through the authenticated distributed
 * runtime protocol. This object exists only so hardware-only nodes can use the
 * common worker lifecycle; it can never advertise or synthesize inference.
 */
export class MycelliosNativeControlAdapter implements InferenceAdapter {
  readonly kind = "mycellios-native" as const;

  async probe(): Promise<AdapterCapabilities> {
    return {
      kind: this.kind,
      models: [],
      streaming: false,
      embeddings: false,
      reranking: false,
    };
  }

  async warm(): Promise<void> {
    throw nativeDeploymentRequired();
  }

  async *generate(_request: AdapterRequest, _signal: AbortSignal): AsyncIterable<AdapterChunk> {
    throw nativeDeploymentRequired();
  }

  async cancel(): Promise<void> {
    // There is no adapter-owned inference job to cancel.
  }

  async metrics(): Promise<AdapterMetrics> {
    return { ready: false, activeJobs: 0, loadedModels: [] };
  }
}

function nativeDeploymentRequired(): AdapterError {
  return new AdapterError(
    "No native Mycellios model deployment is active on this node.",
    "native_deployment_required",
    false,
  );
}
