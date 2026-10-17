import type { BrowserLayerExecute } from "../contracts/browser-layer";
import type { BrowserLayerBackend, BrowserLayerRunner } from "./browser-layer-runner";

interface LayerTask {
  taskId: string;
  leaseId: string;
  artifactId: string;
  deadlineAt: number;
}

export interface LayerLoadTask extends LayerTask {
  manifestUrl: string;
}

export type LayerExecuteTask = LayerTask & BrowserLayerExecute;

export interface BrowserLayerWorkOptions {
  backend(): BrowserLayerBackend;
  token(): string | null;
  signal(): AbortSignal;
  active(): boolean;
  send(type: string, payload: unknown): void;
  status(message: string): void;
}

const SHA256 = /^[0-9a-f]{64}$/;

/** Browser worker handler for independently assigned ONNX stages. */
export class BrowserLayerWork {
  private readonly resident = new Map<string, BrowserLayerRunner>();

  constructor(private readonly options: BrowserLayerWorkOptions) {}

  async load(task: LayerLoadTask): Promise<void> {
    if (!this.available(task)) return;
    let runner: BrowserLayerRunner | undefined;
    try {
      if (!SHA256.test(task.artifactId)
        || task.manifestUrl !== `/mobile/v1/layers/${task.artifactId}/manifest`) {
        throw new Error("untrusted browser layer manifest path");
      }
      this.options.status("Downloading and checking an assigned model layer…");
      const token = this.options.token();
      if (!token) throw new Error("browser worker credentials are unavailable");
      const response = await fetch(new URL(task.manifestUrl, location.origin), {
        signal: this.options.signal(),
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error(`browser layer manifest HTTP ${response.status}`);
      const { browserLayerManifestSchema } = await import("../contracts/browser-layer");
      const manifest = browserLayerManifestSchema.parse(await response.json());
      if (manifest.artifactId !== task.artifactId) {
        throw new Error("browser layer identity mismatch");
      }
      const { BrowserLayerRunner } = await import("./browser-layer-runner");
      const backend = this.options.backend();
      for (const [residentId, previous] of this.resident) {
        if (residentId === task.artifactId) continue;
        this.resident.delete(residentId);
        await previous.release();
      }
      runner = await BrowserLayerRunner.load(manifest, backend, token, this.options.signal());
      const canary = manifest.canary;
      const canaryRequest = crypto.randomUUID();
      const started = performance.now();
      const result = await runner.run({
        requestId: canaryRequest,
        position: 0,
        tokens: canary.tokens,
        hidden: decodeFloat32(canary.hiddenBase64),
        auxiliary: Object.fromEntries(Object.entries(canary.auxiliaryBase64)
          .map(([name, value]) => [name, decodeFloat32(value)])),
      });
      runner.reset(canaryRequest);
      const expected = decodeFloat32(canary.expectedBase64);
      if (!closeFloat32(result.hidden, expected, 2e-4)) {
        throw new Error("browser layer canary differs from native output");
      }
      if (!this.available(task)) {
        await runner.release();
        return;
      }
      await this.resident.get(task.artifactId)?.release();
      this.resident.set(task.artifactId, runner);
      this.options.send("layer.ready", {
        taskId: task.taskId,
        leaseId: task.leaseId,
        artifactId: task.artifactId,
        outputBase64: encodeFloat32(result.hidden),
        backend,
        durationMs: performance.now() - started,
      });
      this.options.status(`Model layer ${manifest.layer} ready on ${backend}.`);
    } catch (error) {
      await runner?.release();
      if (this.options.signal().aborted) return;
      this.options.send("layer.fail", {
        taskId: task.taskId,
        leaseId: task.leaseId,
        artifactId: task.artifactId,
        message: error instanceof Error ? error.message.slice(0, 500) : "browser layer load failed",
      });
    }
  }

  async execute(task: LayerExecuteTask): Promise<void> {
    if (!this.available(task)) return;
    const runner = this.resident.get(task.artifactId);
    try {
      if (!runner) throw new Error("assigned browser layer is not resident");
      const result = await runner.run({
        requestId: task.requestId,
        position: task.position,
        tokens: task.tokens,
        hidden: decodeFloat32(task.hiddenBase64),
        auxiliary: Object.fromEntries(Object.entries(task.auxiliaryBase64)
          .map(([name, value]) => [name, decodeFloat32(value)])),
      });
      if (!this.available(task)) return;
      this.options.send("layer.result", {
        taskId: task.taskId,
        leaseId: task.leaseId,
        artifactId: task.artifactId,
        requestId: task.requestId,
        position: task.position,
        tokens: task.tokens,
        outputBase64: encodeFloat32(result.hidden),
        stateBase64: Object.fromEntries(Object.entries(result.state)
          .map(([name, values]) => [name, encodeFloat32(values)])),
        backend: runner.backend,
        durationMs: result.durationMs,
      });
      this.options.status(`Executed layer ${runner.manifest.layer} for the model network.`);
    } catch (error) {
      runner?.reset(task.requestId);
      if (this.options.signal().aborted) return;
      this.options.send("layer.fail", {
        taskId: task.taskId,
        leaseId: task.leaseId,
        artifactId: task.artifactId,
        message: error instanceof Error ? error.message.slice(0, 500) : "browser layer execution failed",
      });
    }
  }

  reset(artifactId: string, requestId: string): void {
    this.resident.get(artifactId)?.reset(requestId);
  }

  async close(): Promise<void> {
    const runners = [...this.resident.values()];
    this.resident.clear();
    await Promise.allSettled(runners.map((runner) => runner.release()));
  }

  private available(task: LayerTask): boolean {
    return this.options.active() && !this.options.signal().aborted
      && task.deadlineAt > Date.now();
  }
}

function decodeFloat32(base64: string): Float32Array {
  const bytes = Uint8Array.from(atob(base64), (character) => character.charCodeAt(0));
  if (bytes.byteLength % 4 !== 0 || bytes.byteLength > 8 * 1024 * 1024) {
    throw new Error("invalid browser layer tensor encoding");
  }
  return new Float32Array(bytes.buffer);
}

function encodeFloat32(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 16_384) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 16_384));
  }
  return btoa(binary);
}

function closeFloat32(actual: Float32Array, expected: Float32Array, tolerance: number): boolean {
  return actual.length === expected.length && actual.every((value, index) =>
    Number.isFinite(value) && Math.abs(value - (expected[index] ?? Infinity)) <= tolerance);
}
