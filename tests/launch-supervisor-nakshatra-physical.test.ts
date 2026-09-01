import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
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

// Archived research fixture. The product launcher no longer accepts this
// backend, so the historical physical harness must never enter the test gate.
const physicalIt = it.skip;

interface NakshatraPackageDocument {
  packageId: string;
  model: {
    source: string;
    revision: string | null;
  };
  stage: {
    layerStart: number;
    layerEnd: number;
    totalLayers: number;
  };
  compatibility: {
    maxContextTokens: number;
  };
}

describe("physical Nakshatra gdlp-python-launch/2 route", () => {
  physicalIt(
    "launches the sealed sub-GGUF final stage and serves a real completion",
    async () => {
      const workspace = process.cwd();
      const pythonExecutable = resolve(
        process.env.GDLP_NAKSHATRA_PYTHON ??
          resolve(workspace, "runtime", "distribution-venv", "Scripts", "python.exe"),
      );
      const snapshot = resolve(
        process.env.GDLP_NAKSHATRA_MODEL_SNAPSHOT ??
          resolve(
            workspace,
            "runtime",
            "hf-cache",
            "hub",
            "models--HuggingFaceTB--SmolLM2-135M-Instruct",
            "snapshots",
            "12fd25f77366fa6b3b4b768ec3050bf629380bac",
          ),
      );
      const packagePath = resolve(
        process.env.GDLP_NAKSHATRA_PACKAGE ??
          resolve(
            workspace,
            "runtime",
            "packages",
            "smollm2-135m-f16-nakshatra-l15-30",
          ),
      );
      const configuredCpuDaemon = resolve(
        workspace,
        "runtime",
        "external",
        "nakshatra-stage",
        "llama.cpp",
        "build-gdlp-nakshatra-cpu",
        "bin",
        "llama-nakshatra-worker.exe",
      );
      const legacyCpuDaemon = resolve(
        workspace,
        "runtime",
        "external",
        "nakshatra-stage",
        "llama.cpp",
        "build-gdlp-nakshatra",
        "bin",
        "llama-nakshatra-worker.exe",
      );
      const daemonExecutable = resolve(
        process.env.GDLP_NAKSHATRA_DAEMON ??
          (existsSync(configuredCpuDaemon) ? configuredCpuDaemon : legacyCpuDaemon),
      );
      await Promise.all([
        access(pythonExecutable),
        access(snapshot),
        access(packagePath),
        access(daemonExecutable),
      ]);

      const packageManifestBytes = await readFile(
        resolve(packagePath, "nakshatra-stage.json"),
      );
      const packageManifestText = packageManifestBytes.toString("utf8");
      const packageDocument = JSON.parse(packageManifestText) as NakshatraPackageDocument;
      const pipelineMatch = packageManifestText.match(/"pipelineId":(\d+)/);
      if (!pipelineMatch) throw new Error("nakshatra_package_pipeline_id_is_missing");
      const pipelineId = pipelineMatch[1]!;
      const manifestSha256 = createHash("sha256")
        .update(packageManifestBytes)
        .digest("hex");
      const profileEnvelope = JSON.parse(
        await readFile(
          resolve(workspace, "docs", "benchmarks", "smollm2-135m-model-profile.json"),
          "utf8",
        ),
      ) as { model: DistributedModelProfile };
      const stageRange = packageDocument.stage;
      expect(stageRange).toMatchObject({ layerStart: 15, layerEnd: 30, totalLayers: 30 });

      const ports = await reservePorts(4);
      const nodes = [
        node("nakshatra-root", ports[0]!, "python-transformers"),
        node("nakshatra-stage", ports[1]!, "nakshatra-llama.cpp"),
      ];
      const plan: DistributionPlan = {
        algorithm: "physical-nakshatra-launch-smoke",
        codec: "fp16",
        microBatchSize: 1,
        prefillChunkTokens: 32,
        stages: [
          { nodeId: nodes[0]!.id, layerStart: 0, layerEnd: stageRange.layerStart },
          {
            nodeId: nodes[1]!.id,
            layerStart: stageRange.layerStart,
            layerEnd: stageRange.layerEnd,
          },
        ],
      };
      const request: RuntimePlanRequest = {
        model: profileEnvelope.model,
        modelRevision: packageDocument.model.revision ?? "nakshatra-local",
        tokenizerId: "HuggingFaceTB/SmolLM2-135M-Instruct",
        topology: {
          nodes,
          links: [link(nodes[0]!.id, nodes[1]!.id), link(nodes[1]!.id, nodes[0]!.id)],
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
      const manifest = buildRuntimePipelineManifest(request);
      const stageId = manifest.plans.prefill.stages[1]!.stageId;
      const description = compilePythonLaunchDescription(manifest, {
        apiEndpoint: { host: "127.0.0.1", port: ports[3]! },
        returnEndpoint: { host: "127.0.0.1", port: ports[2]! },
        returnBindHost: "127.0.0.1",
        runtimeModel: { source: snapshot, revision: null, snapshotIdentity: pipelineId },
        publicModelName: "distributed-nakshatra-small",
        pythonExecutable,
        threadsPerStage: 1,
        connectTimeoutSeconds: 60,
        batchWindowMs: 0,
        maxPendingRequests: 4,
        maxOutputTokens: 8,
        nakshatraStages: {
          [stageId]: {
            packagePath,
            packageId: packageDocument.packageId,
            manifestSha256,
            modelSource: packageDocument.model.source,
            modelRevision: packageDocument.model.revision,
            layerStart: stageRange.layerStart,
            layerEnd: stageRange.layerEnd,
            totalLayers: stageRange.totalLayers,
            daemonExecutable,
            pipelineId,
            contextTokens: Math.min(128, packageDocument.compatibility.maxContextTokens),
            gpuLayers: 0,
            computeApi: "cpu",
            startupTimeoutSeconds: 60,
            callTimeoutSeconds: 60,
            closeTimeoutSeconds: 10,
          },
        },
      } as unknown as Parameters<typeof compilePythonLaunchDescription>[1]);
      expect(description.launchOrder[0]).toMatchObject({
        kind: "remote-stage",
        stageId,
        nakshatra: { packageId: packageDocument.packageId, computeApi: "cpu" },
      });

      const localAgent = new LocalProcessAgent({
        id: "physical-nakshatra-local-agent",
        cwd: workspace,
        allowedExecutables: [description.configuration.pythonExecutable],
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
        readinessTimeoutMs: 60_000,
      });

      let completed = false;
      try {
        const launchStartedAt = performance.now();
        const running = await supervisor.start();
        const launchMs = performance.now() - launchStartedAt;
        expect(running.state).toBe("running");

        const responseStartedAt = performance.now();
        const response = await fetch(`http://127.0.0.1:${ports[3]!}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "distributed-nakshatra-small",
            messages: [{ role: "user", content: "The capital of France is" }],
            max_tokens: 4,
            temperature: 0,
          }),
        });
        const completion = await response.json() as {
          object?: string;
          usage?: { completion_tokens?: number };
          distribution_metrics?: { ttft_ms?: number; tpot_ms?: number; pipeline_ms?: number };
        };
        const requestWallMs = performance.now() - responseStartedAt;
        expect(response.status).toBe(200);
        expect(completion.object).toBe("chat.completion");
        expect(completion.usage?.completion_tokens).toBeGreaterThan(0);
        expect(completion.distribution_metrics?.pipeline_ms).toBeGreaterThan(0);
        process.stdout.write(
          `GDLP_PHYSICAL_NAKSHATRA_LAUNCH_SMOKE ${JSON.stringify({
            schema: "gdlp-physical-nakshatra-launch-smoke/1",
            stages: 2,
            boundaries: [0, stageRange.layerStart, stageRange.layerEnd],
            codec: "fp16",
            packageId: packageDocument.packageId,
            launchId: description.launchId,
            launchMs,
            requestWallMs,
            completionTokens: completion.usage?.completion_tokens,
            metrics: completion.distribution_metrics,
          })}\n`,
        );
        completed = true;
      } catch (error) {
        throw new Error(
          `physical_nakshatra_launcher_failed:${error instanceof Error ? error.message : String(error)}\n${JSON.stringify(supervisor.snapshot(), null, 2)}`,
          { cause: error },
        );
      } finally {
        const stopped = await supervisor.stop("physical_nakshatra_test_complete");
        if (completed) {
          expect(stopped.state).toBe("stopped");
          expect(stopped.processes.every((process) => process.state === "stopped")).toBe(true);
        }
      }
    },
    180_000,
  );
});

function node(id: string, port: number, engine: string) {
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
      engine,
      version: "physical",
      modelFormats: engine === "nakshatra-llama.cpp" ? ["sub-gguf-nakshatra"] : ["safetensors"],
      executionModes: ["layer-range"],
    },
    capabilities: {
      deviceKinds: ["cpu"],
      computeApis: engine === "nakshatra-llama.cpp" ? ["llama.cpp", "cpu"] : ["torch-cpu"],
      weightDtypes: engine === "nakshatra-llama.cpp" ? ["f16"] : ["fp32"],
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
