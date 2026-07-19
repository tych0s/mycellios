import { access, readFile } from "node:fs/promises";
import { createServer } from "node:net";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  LocalProcessAgent,
  PythonLaunchSupervisor,
} from "../src/distribution/launch-supervisor.js";
import { compilePythonLaunchDescription } from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
} from "../src/distribution/types.js";

const physicalIt = process.env.RUN_PHYSICAL_LAUNCHER_TESTS === "1" ? it : it.skip;

describe("physical GDLP/2 launch supervisor", () => {
  physicalIt(
    "materializes the compiled Python pipeline and serves a real chat completion",
    async () => {
      const workspace = process.cwd();
      const pythonExecutable = resolve(
        workspace,
        "runtime",
        "distribution-venv",
        "Scripts",
        "python.exe",
      );
      const snapshot = resolve(
        workspace,
        "runtime",
        "hf-cache",
        "hub",
        "models--HuggingFaceTB--SmolLM2-135M-Instruct",
        "snapshots",
        "12fd25f77366fa6b3b4b768ec3050bf629380bac",
      );
      await access(pythonExecutable);
      await access(snapshot);
      const profileEnvelope = JSON.parse(
        await readFile(
          resolve(workspace, "docs", "benchmarks", "smollm2-135m-model-profile.json"),
          "utf8",
        ),
      ) as { model: DistributedModelProfile };
      const ports = await reservePorts(4);
      const rootAnchorPort = ports[0]!;
      const stagePort = ports[1]!;
      const returnPort = ports[2]!;
      const apiPort = ports[3]!;
      const plan: DistributionPlan = {
        algorithm: "physical-launch-smoke",
        codec: "fp16",
        microBatchSize: 1,
        prefillChunkTokens: 32,
        stages: [
          { nodeId: "physical-root", layerStart: 0, layerEnd: 15 },
          { nodeId: "physical-stage-1", layerStart: 15, layerEnd: 30 },
        ],
      };
      const nodes = [
        node("physical-root", rootAnchorPort),
        node("physical-stage-1", stagePort),
      ];
      const request: RuntimePlanRequest = {
        model: profileEnvelope.model,
        modelRevision: "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        tokenizerId: "HuggingFaceTB/SmolLM2-135M-Instruct",
        topology: {
          nodes,
          links: [
            link(nodes[0]!.id, nodes[1]!.id),
            link(nodes[1]!.id, nodes[0]!.id),
          ],
        },
        workload: {
          promptTokens: 32,
          outputTokens: 4,
          contextTokens: 128,
          concurrentSequences: 1,
          maxStages: 2,
          maxQualityLoss: 0,
          minRouteAvailability: 0.9,
          batchWindowMs: 0,
          p95: false,
        },
        phasePlans: { prefill: plan, decode: plan },
      };
      const description = compilePythonLaunchDescription(
        buildRuntimePipelineManifest(request),
        {
          apiEndpoint: { host: "127.0.0.1", port: apiPort },
          returnEndpoint: { host: "127.0.0.1", port: returnPort },
          returnBindHost: "127.0.0.1",
          runtimeModel: { source: snapshot, revision: null },
          publicModelName: "distributed-small",
          pythonExecutable,
          threadsPerStage: 1,
          connectTimeoutSeconds: 45,
          batchWindowMs: 0,
          maxPendingRequests: 8,
          maxOutputTokens: 8,
        },
      );
      const localAgent = new LocalProcessAgent({
        id: "physical-local-agent",
        cwd: workspace,
        env: {
          PYTHONPATH: resolve(workspace, "python"),
          HF_HOME: resolve(workspace, "runtime", "hf-cache"),
          TOKENIZERS_PARALLELISM: "false",
        },
        maxOutputBytesPerStream: 256 * 1024,
        stopGraceMs: 20_000,
      });
      const supervisor = new PythonLaunchSupervisor(description, {
        resolveAgent: () => localAgent,
        readinessTimeoutMs: 45_000,
      });

      let completed = false;
      try {
        const launchStartedAt = performance.now();
        const running = await supervisor.start();
        const launchMs = performance.now() - launchStartedAt;
        expect(running.state).toBe("running");
        expect(running.processes.every((process) => process.state === "ready")).toBe(true);

        const health = await fetch(`http://127.0.0.1:${apiPort}/health`);
        expect(health.status).toBe(200);
        expect(await health.json()).toMatchObject({
          status: "ready",
          model: "distributed-small",
          stages: 2,
          boundaries: [0, 15, 30],
          codec: "fp16",
        });

        const requestStartedAt = performance.now();
        const response = await fetch(`http://127.0.0.1:${apiPort}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "distributed-small",
            messages: [{ role: "user", content: "Responde solamente: hola" }],
            max_tokens: 4,
            temperature: 0,
          }),
        });
        const completion = await response.json() as {
          object?: string;
          choices?: Array<{ message?: { content?: string } }>;
          usage?: { completion_tokens?: number };
          distribution_metrics?: { ttft_ms?: number; tpot_ms?: number; pipeline_ms?: number };
        };
        const requestWallMs = performance.now() - requestStartedAt;
        expect(response.status).toBe(200);
        expect(completion.object).toBe("chat.completion");
        expect(completion.choices?.[0]?.message?.content).toBeTypeOf("string");
        expect(completion.usage?.completion_tokens).toBeGreaterThan(0);
        expect(completion.distribution_metrics?.pipeline_ms).toBeGreaterThan(0);
        process.stdout.write(
          `GDLP_PHYSICAL_LAUNCH_SMOKE ${JSON.stringify({
            schema: "gdlp-physical-launch-smoke/1",
            stages: 2,
            boundaries: [0, 15, 30],
            codec: "fp16",
            launchMs,
            requestWallMs,
            completionTokens: completion.usage?.completion_tokens,
            metrics: completion.distribution_metrics,
          })}\n`,
        );
        completed = true;
      } catch (error) {
        const snapshot = supervisor.snapshot();
        throw new Error(
          `physical_launcher_failed:${error instanceof Error ? error.message : String(error)}\n${JSON.stringify(snapshot, null, 2)}`,
          { cause: error },
        );
      } finally {
        const stopped = await supervisor.stop("physical_test_complete");
        if (completed) {
          expect(stopped.state).toBe("stopped");
          expect(stopped.processes.every((process) => process.state === "stopped")).toBe(true);
        }
      }
    },
    180_000,
  );
});

function node(id: string, port: number) {
  return {
    id,
    region: "physical-local",
    memoryBytes: 768 * 1024 * 1024,
    reserveBytes: 64 * 1024 * 1024,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0,
    maxBatchSpeedup: 1,
    powerWatts: 65,
    availability: 0.999,
    endpoint: { host: "127.0.0.1", port },
    backend: {
      engine: "python-transformers",
      version: "1",
      modelFormats: ["safetensors"],
      executionModes: ["layer-range"],
    },
    capabilities: {
      deviceKinds: ["cpu"],
      computeApis: ["torch-cpu"],
      weightDtypes: ["fp32"],
      activationCodecs: ["fp16" as const],
      features: ["layer-range", "kv-reuse"],
    },
  };
}

function link(from: string, to: string) {
  return {
    from,
    to,
    oneWayLatencyMs: 0.1,
    jitterP95Ms: 0,
    bandwidthMbps: 10_000,
    lossRate: 0,
    availability: 0.999,
  };
}

async function reservePorts(count: number): Promise<number[]> {
  const servers = [];
  try {
    for (let index = 0; index < count; index += 1) {
      const server = createServer();
      await new Promise<void>((resolveReady, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolveReady);
      });
      servers.push(server);
    }
    return servers.map((server) => {
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("port_reservation_failed");
      return address.port;
    });
  } finally {
    await Promise.all(
      servers.map(
        (server) => new Promise<void>((resolveClosed) => server.close(() => resolveClosed())),
      ),
    );
  }
}
