import * as ort from "onnxruntime-web/webgpu";
import { browserLayerManifestSchema, tensorElements, tensorShape,
  type BrowserLayerManifest } from "../contracts/browser-layer";

export type BrowserLayerBackend = "webgpu" | "cpu";

export interface BrowserLayerInput {
  requestId: string;
  position: number;
  tokens: number;
  hidden: Float32Array;
  auxiliary: Record<string, Float32Array>;
}

export interface BrowserLayerOutput {
  hidden: Float32Array;
  state: Record<string, Float32Array>;
  durationMs: number;
}

interface RequestCache {
  state: Map<string, ort.Tensor>;
  nextPosition: number;
  updatedAt: number;
}

/** Executes any conforming float32 ONNX stage without model-specific tensor names. */
export class BrowserLayerRunner {
  private readonly requests = new Map<string, RequestCache>();
  private readonly inFlight = new Map<string, Promise<BrowserLayerOutput>>();
  private readonly deferredReset = new Set<string>();

  private constructor(
    readonly manifest: BrowserLayerManifest,
    readonly backend: BrowserLayerBackend,
    private readonly session: ort.InferenceSession,
  ) {}

  static async load(
    candidate: BrowserLayerManifest, backend: BrowserLayerBackend,
    token: string, signal?: AbortSignal,
  ): Promise<BrowserLayerRunner> {
    const manifest = browserLayerManifestSchema.parse(candidate);
    if (backend === "webgpu" && !navigator.gpu) throw new Error("WebGPU is unavailable");
    const url = new URL(`/mobile/v1/layers/graphs/${manifest.graphSha256}`, location.origin);
    const response = await fetch(url, {
      ...(signal ? { signal } : {}),
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) throw new Error(`browser layer graph HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (bytes.byteLength < 1 || bytes.byteLength > 512 * 1024 * 1024) {
      throw new Error("browser layer graph exceeds the supported size");
    }
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)))
      .map((byte) => byte.toString(16).padStart(2, "0")).join("");
    if (digest !== manifest.graphSha256) throw new Error("browser layer graph hash mismatch");
    ort.env.wasm.wasmPaths = "/browser/ort/";
    ort.env.wasm.numThreads = 1;
    const stateLocations = Object.fromEntries(manifest.state.map((item) =>
      [item.outputName, "gpu-buffer" as const]));
    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: [backend === "webgpu" ? "webgpu" : "wasm"],
      ...(backend === "webgpu" && manifest.state.length
        ? { preferredOutputLocation: stateLocations } : {}),
    });
    const requiredInputs = [manifest.hidden.inputName, ...manifest.auxiliary.map((item) => item.name),
      ...manifest.state.map((item) => item.inputName)];
    const requiredOutputs = [manifest.hidden.outputName, ...manifest.state.flatMap((item) =>
      [item.outputName, item.deltaOutputName])];
    if (requiredInputs.some((name) => !session.inputNames.includes(name))
      || requiredOutputs.some((name) => !session.outputNames.includes(name))) {
      await session.release();
      throw new Error("browser layer graph does not implement its tensor manifest");
    }
    return new BrowserLayerRunner(manifest, backend, session);
  }

  async run(input: BrowserLayerInput): Promise<BrowserLayerOutput> {
    if (this.inFlight.has(input.requestId)) throw new Error("browser layer request is already running");
    const operation = this.runInternal(input);
    this.inFlight.set(input.requestId, operation);
    try { return await operation; } finally {
      this.inFlight.delete(input.requestId);
      if (this.deferredReset.delete(input.requestId)) this.reset(input.requestId);
    }
  }

  private async runInternal(input: BrowserLayerInput): Promise<BrowserLayerOutput> {
    const { requestId, position, tokens } = input;
    const { manifest } = this;
    if (!requestId || !Number.isSafeInteger(position) || position < 0
      || !Number.isSafeInteger(tokens) || tokens < 1
      || position + tokens > manifest.maxContextTokens
      || input.hidden.length !== tokens * manifest.hidden.width
      || Object.keys(input.auxiliary).sort().join("|")
        !== manifest.auxiliary.map((item) => item.name).sort().join("|")) {
      throw new Error("browser layer input shape is invalid");
    }
    for (const item of manifest.auxiliary) {
      const values = input.auxiliary[item.name];
      if (!values || values.length !== tensorElements(item.shape, tokens, position)) {
        throw new Error("browser layer auxiliary tensor shape is invalid");
      }
    }
    for (const [staleId, stored] of this.requests) {
      if (Date.now() - stored.updatedAt > 5 * 60_000) this.reset(staleId);
    }
    let cache = this.requests.get(requestId);
    if (!cache) {
      if (position !== 0) throw new Error("browser layer request has no initial state");
      if (this.requests.size >= 8) throw new Error("browser layer state capacity reached");
      cache = { state: new Map(manifest.state.map((item) => [item.name,
        new ort.Tensor("float32", new Float32Array(0), tensorShape(item.shape, tokens, 0))])),
        nextPosition: 0, updatedAt: Date.now() };
      this.requests.set(requestId, cache);
    }
    if (cache.nextPosition !== position) throw new Error("browser layer sequence is out of order");
    const feeds: Record<string, ort.Tensor> = {
      [manifest.hidden.inputName]: new ort.Tensor("float32", input.hidden,
        [1, tokens, manifest.hidden.width]),
    };
    for (const item of manifest.auxiliary) {
      feeds[item.name] = new ort.Tensor("float32", input.auxiliary[item.name]!,
        tensorShape(item.shape, tokens, position));
    }
    for (const item of manifest.state) feeds[item.inputName] = cache.state.get(item.name)!;
    const started = performance.now();
    let outputs: ort.InferenceSession.OnnxValueMapType;
    try { outputs = await this.session.run(feeds); } catch (error) {
      this.reset(requestId);
      throw error;
    }
    try {
      const output = outputs[manifest.hidden.outputName];
      if (!output) throw new Error("browser layer graph omitted hidden output");
      const hidden = new Float32Array(await output.getData() as Float32Array);
      if (hidden.length !== tokens * manifest.hidden.width || !hidden.every(Number.isFinite)) {
        throw new Error("browser layer hidden output is invalid");
      }
      const state: Record<string, Float32Array> = {};
      const nextState = new Map<string, ort.Tensor>();
      for (const item of manifest.state) {
        const full = outputs[item.outputName];
        const delta = outputs[item.deltaOutputName];
        if (!full || !delta) throw new Error("browser layer graph omitted state output");
        const values = new Float32Array(await delta.getData() as Float32Array);
        if (values.length !== tensorElements(item.shape, tokens, tokens)
          || full.size !== tensorElements(item.shape, tokens, position + tokens)
          || !values.every(Number.isFinite)
          || (this.backend === "webgpu" && full.location !== "gpu-buffer")) {
          throw new Error("browser layer state output or backend is invalid");
        }
        state[item.name] = values;
        nextState.set(item.name, full);
      }
      for (const tensor of cache.state.values()) tensor.dispose();
      cache.state = nextState;
      cache.nextPosition += tokens;
      cache.updatedAt = Date.now();
      for (const [name, tensor] of Object.entries(outputs)) {
        if (![...nextState.values()].includes(tensor)) tensor.dispose();
      }
      return { hidden, state, durationMs: performance.now() - started };
    } catch (error) {
      for (const tensor of Object.values(outputs)) tensor.dispose();
      this.reset(requestId);
      throw error;
    }
  }

  reset(requestId: string): void {
    if (this.inFlight.has(requestId)) { this.deferredReset.add(requestId); return; }
    const cache = this.requests.get(requestId);
    if (!cache) return;
    for (const tensor of cache.state.values()) tensor.dispose();
    this.requests.delete(requestId);
  }

  async release(): Promise<void> {
    await Promise.allSettled([...this.inFlight.values()]);
    for (const requestId of this.requests.keys()) this.reset(requestId);
    await this.session.release();
  }
}
