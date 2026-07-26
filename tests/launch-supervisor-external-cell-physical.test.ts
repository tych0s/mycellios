import { execFileSync } from "node:child_process";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  LocalProcessAgent,
  PythonLaunchSupervisor,
} from "../src/distribution/launch-supervisor.js";
import { compilePythonLaunchDescription } from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimeNodeProfile,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
} from "../src/distribution/types.js";

const physicalIt =
  process.env.RUN_PHYSICAL_EXTERNAL_CELL_TESTS === "1" ? it : it.skip;
const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const SNAPSHOT_COMMIT = "12fd25f77366fa6b3b4b768ec3050bf629380bac";
const CELL_LAYER_START = 14;
const CELL_LAYER_END = 16;
const PLANNED_CONTEXT_TOKENS = 128;

interface FixtureCompilation {
  destination: string;
  schema: "gdlp-llama-cell-stage/2";
  layer_start: number;
  layer_end: number;
  world_size: number;
  rank_weights: number[];
  manifest_sha256: string;
  shard_sha256: string[];
  rank_fixed_bytes: number[];
  rank_kv_bytes_per_token: number[];
}

describe("physical external TP-cell launch supervisor", () => {
  physicalIt(
    "launches final stage, rank-1 member, rank-0 anchor and root as separate processes",
    async () => {
      const workspace = process.cwd();
      const pythonExecutable = resolve(
        workspace,
        "runtime",
        "distribution-venv",
        "Scripts",
        "python.exe",
      );
      const snapshot =
        process.env.GDLP_PHYSICAL_MODEL_SNAPSHOT ??
        resolve(
          workspace,
          "runtime",
          "hf-cache",
          "hub",
          "models--HuggingFaceTB--SmolLM2-135M-Instruct",
          "snapshots",
          SNAPSHOT_COMMIT,
        );
      await access(pythonExecutable);
      await access(snapshot);
      const profileEnvelope = JSON.parse(
        await readFile(
          resolve(workspace, "docs", "benchmarks", "smollm2-135m-model-profile.json"),
          "utf8",
        ),
      ) as { model: DistributedModelProfile };
      const temporaryRoot = await mkdtemp(join(tmpdir(), "gdlp-external-cell-e2e-"));
      const fixturePath = join(temporaryRoot, "layers-14-16-world-2");
      const childEnvironment = {
        ...process.env,
        PYTHONPATH: resolve(workspace, "python"),
        HF_HOME: resolve(workspace, "runtime", "hf-cache"),
        TOKENIZERS_PARALLELISM: "false",
      };

      let supervisor: PythonLaunchSupervisor | undefined;
      let completed = false;
      try {
        const fixture = compileFixture(
          pythonExecutable,
          snapshot,
          fixturePath,
          workspace,
          childEnvironment,
        );
        expect(fixture).toMatchObject({
          destination: fixturePath,
          schema: "gdlp-llama-cell-stage/2",
          layer_start: CELL_LAYER_START,
          layer_end: CELL_LAYER_END,
          world_size: 2,
        });
        expect(fixture.shard_sha256).toHaveLength(2);
        expect(fixture.rank_fixed_bytes).toHaveLength(2);
        expect(fixture.rank_kv_bytes_per_token).toHaveLength(2);

        const snapshotIdentity = readSnapshotIdentity(
          pythonExecutable,
          snapshot,
          workspace,
          childEnvironment,
        );
        const ports = await reservePorts(8);
        const [
          rootAnchorPort,
          cellAnchorPort,
          cellPeerPort,
          finalStagePort,
          returnPort,
          apiPort,
          controlPort,
          distributedPort,
        ] = ports as [number, number, number, number, number, number, number, number];
        const plan: DistributionPlan = {
          algorithm: "physical-external-cell-smoke",
          codec: "fp16",
          microBatchSize: 1,
          prefillChunkTokens: 32,
          stages: [
            { nodeId: "physical-root", layerStart: 0, layerEnd: CELL_LAYER_START },
            {
              nodeId: "physical-cell-anchor",
              layerStart: CELL_LAYER_START,
              layerEnd: CELL_LAYER_END,
            },
            {
              nodeId: "physical-final",
              layerStart: CELL_LAYER_END,
              layerEnd: profileEnvelope.model.layers.length,
            },
          ],
        };
        const nodes = [
          pipelineNode("physical-root", rootAnchorPort),
          cellNode("physical-cell-anchor", cellAnchorPort),
          cellNode("physical-cell-peer", cellPeerPort),
          pipelineNode("physical-final", finalStagePort),
        ];
        const rankMemory = fixture.rank_fixed_bytes.map((fixedBytes, rank) => {
          const kvBytesPerToken = fixture.rank_kv_bytes_per_token[rank]!;
          return {
            fixedBytes,
            kvBytesPerToken,
            requiredBytes: fixedBytes + kvBytesPerToken * PLANNED_CONTEXT_TOKENS,
          };
        });
        const request: RuntimePlanRequest = {
          model: profileEnvelope.model,
          modelRevision: SNAPSHOT_COMMIT,
          tokenizerId: "HuggingFaceTB/SmolLM2-135M-Instruct",
          topology: {
            nodes,
            links: completeLinks(nodes),
          },
          workload: {
            promptTokens: 32,
            outputTokens: 4,
            contextTokens: PLANNED_CONTEXT_TOKENS,
            concurrentSequences: 1,
            maxStages: 3,
            maxQualityLoss: 0,
            minRouteAvailability: 0.9,
            batchWindowMs: 0,
            p95: false,
          },
          phasePlans: { prefill: plan, decode: plan },
          tensorParallelCells: [
            {
              stageIndex: 1,
              memberNodeIds: ["physical-cell-anchor", "physical-cell-peer"],
              execution: {
                mode: "tensor-parallel-cell",
                engine: "python-torch",
                collectiveBackend: "gloo",
                computeDtype: "float32",
                fixture: {
                  schema: fixture.schema,
                  location: "member-local",
                  path: fixturePath,
                  layerCount: CELL_LAYER_END - CELL_LAYER_START,
                  manifestSha256: fixture.manifest_sha256,
                  shardSha256: fixture.shard_sha256,
                  rankMemory,
                },
                worldSize: 2,
                rankMemberIds: ["physical-cell-anchor", "physical-cell-peer"],
                rankWeights: fixture.rank_weights,
                rankDevices: ["cpu", "cpu"],
                operationTimeoutSeconds: 120,
                external: {
                  rankFixturePaths: [fixturePath, fixturePath],
                  controlBindHost: "127.0.0.1",
                  controlAdvertiseHost: "127.0.0.1",
                  controlPort,
                  distributedAdvertiseHost: "127.0.0.1",
                  distributedPort,
                  startupTimeoutSeconds: 120,
                },
              },
            },
          ],
        };
        const description = compilePythonLaunchDescription(
          buildRuntimePipelineManifest(request),
          {
            apiEndpoint: { host: "127.0.0.1", port: apiPort },
            returnEndpoint: { host: "127.0.0.1", port: returnPort },
            returnBindHost: "127.0.0.1",
            runtimeModel: {
              source: snapshot,
              revision: null,
              snapshotIdentity,
            },
            publicModelName: "distributed-external-cell",
            pythonExecutable,
            threadsPerStage: 1,
            connectTimeoutSeconds: 120,
            batchWindowMs: 0,
            maxPendingRequests: 8,
            maxOutputTokens: 8,
          },
        );
        expect(description.launchOrder.map((entry) => entry.kind)).toEqual([
          "remote-stage",
          "cell-member",
          "remote-stage",
          "root-engine",
        ]);
        expect(description.launchOrder.map((entry) => entry.stageIndex)).toEqual([
          2, 1, 1, 0,
        ]);

        const localAgent = new LocalProcessAgent({
          id: "physical-external-cell-local-agent",
          cwd: workspace,
          allowedExecutables: [description.configuration.pythonExecutable],
          env: childEnvironment,
          maxOutputBytesPerStream: 512 * 1024,
          stopGraceMs: 20_000,
        });
        supervisor = new PythonLaunchSupervisor(description, {
          resolveAgent: () => localAgent,
          readinessTimeoutMs: 120_000,
        });

        const launchStartedAt = performance.now();
        const running = await supervisor.start();
        const launchMs = performance.now() - launchStartedAt;
        expect(running.state).toBe("running");
        expect(running.processes).toHaveLength(4);
        expect(running.processes.every((process) => process.state === "ready")).toBe(true);
        expect(
          running.telemetry
            .filter((event) => event.type === "process_ready")
            .map((event) => event.processId),
        ).toEqual(description.launchOrder.map((entry) => entry.processId));

        const health = await fetch(`http://127.0.0.1:${apiPort}/health`);
        expect(health.status).toBe(200);
        expect(await health.json()).toMatchObject({
          status: "ready",
          model: "distributed-external-cell",
          stages: 3,
          boundaries: [0, CELL_LAYER_START, CELL_LAYER_END, 30],
          codec: "fp16",
        });

        const requestStartedAt = performance.now();
        const response = await fetch(`http://127.0.0.1:${apiPort}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "distributed-external-cell",
            messages: [{ role: "user", content: "Responde solamente: hola" }],
            max_tokens: 3,
            temperature: 0,
          }),
        });
        const completion = (await response.json()) as {
          object?: string;
          choices?: Array<{ message?: { content?: string } }>;
          usage?: { completion_tokens?: number };
          distribution_metrics?: {
            pipeline_ms?: number;
            ttft_ms?: number;
            tpot_ms?: number;
          };
        };
        const requestWallMs = performance.now() - requestStartedAt;
        expect(response.status).toBe(200);
        expect(completion.object).toBe("chat.completion");
        expect(completion.choices?.[0]?.message?.content).toBeTypeOf("string");
        expect(completion.usage?.completion_tokens).toBeGreaterThan(0);
        expect(completion.distribution_metrics?.pipeline_ms).toBeGreaterThan(0);

        const beforeStop = supervisor.snapshot();
        expect(
          beforeStop.processes.find((process) => process.kind === "cell-member")?.output
            ?.stderr,
        ).toContain('"event": "joining_external_cell"');
        expect(
          beforeStop.processes.find(
            (process) => process.kind === "remote-stage" && process.stageIndex === 1,
          )?.output?.stderr,
        ).toContain('"loader": "tensor-parallel-cell-external-safetensors-gloo"');
        const stopped = await supervisor.stop("physical_external_cell_test_complete");
        expect(stopped.state).toBe("stopped");
        expect(stopped.processes.every((process) => process.state === "stopped")).toBe(true);
        expect(
          stopped.telemetry
            .filter((event) => event.type === "process_stop_starting")
            .map((event) => event.processId),
        ).toEqual(description.launchOrder.toReversed().map((entry) => entry.processId));
        supervisor = undefined;

        process.stdout.write(
          `GDLP_PHYSICAL_EXTERNAL_CELL_SMOKE ${JSON.stringify({
            schema: "gdlp-physical-external-cell-smoke/1",
            model: "HuggingFaceTB/SmolLM2-135M-Instruct",
            processes: 4,
            stages: 3,
            cellLayers: [CELL_LAYER_START, CELL_LAYER_END],
            worldSize: 2,
            launchMs,
            requestWallMs,
            completionTokens: completion.usage?.completion_tokens,
            metrics: completion.distribution_metrics,
          })}\n`,
        );
        completed = true;
      } catch (error) {
        const launchSnapshot = supervisor?.snapshot();
        throw new Error(
          `physical_external_cell_launcher_failed:${
            error instanceof Error ? error.message : String(error)
          }\n${JSON.stringify(launchSnapshot, null, 2)}`,
          { cause: error },
        );
      } finally {
        if (supervisor) {
          const stopped = await supervisor.stop("physical_external_cell_test_cleanup");
          if (completed) expect(stopped.state).toBe("stopped");
        }
        await rm(temporaryRoot, { recursive: true, force: true });
      }
    },
    300_000,
  );
});

function compileFixture(
  pythonExecutable: string,
  snapshot: string,
  fixturePath: string,
  workspace: string,
  env: NodeJS.ProcessEnv,
): FixtureCompilation {
  const output = execFileSync(
    pythonExecutable,
    [
      "-m",
      "distributed_runtime.cell_fixture_compiler",
      snapshot,
      fixturePath,
      "--layer-start",
      String(CELL_LAYER_START),
      "--layer-end",
      String(CELL_LAYER_END),
      "--world-size",
      "2",
    ],
    { cwd: workspace, env, encoding: "utf8", maxBuffer: 8 * MIB },
  );
  return JSON.parse(output) as FixtureCompilation;
}

function readSnapshotIdentity(
  pythonExecutable: string,
  snapshot: string,
  workspace: string,
  env: NodeJS.ProcessEnv,
): string {
  const identity = execFileSync(
    pythonExecutable,
    [
      "-c",
      "import sys; from distributed_runtime.model import model_snapshot_identity; print(model_snapshot_identity(sys.argv[1]))",
      snapshot,
    ],
    { cwd: workspace, env, encoding: "utf8", maxBuffer: MIB },
  ).trim();
  if (!/^(?:0|[1-9]\d*)$/.test(identity)) {
    throw new Error(`invalid_model_snapshot_identity:${identity}`);
  }
  return identity;
}

function pipelineNode(id: string, port: number): RuntimeNodeProfile {
  return {
    id,
    region: "physical-local",
    memoryBytes: GIB,
    reserveBytes: 64 * MIB,
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
      activationCodecs: ["fp16"],
      features: ["layer-range", "kv-reuse"],
    },
  };
}

function cellNode(id: string, port: number): RuntimeNodeProfile {
  return {
    ...pipelineNode(id, port),
    backend: {
      engine: "python-torch",
      version: "2",
      modelFormats: ["safetensors"],
      executionModes: ["tensor-parallel-cell"],
    },
    capabilities: {
      deviceKinds: ["cpu"],
      computeApis: ["gloo"],
      weightDtypes: ["fp32"],
      activationCodecs: ["fp16"],
      features: ["rank-local-kv"],
    },
  };
}

function completeLinks(nodes: RuntimeNodeProfile[]) {
  return nodes.flatMap((from) =>
    nodes
      .filter((to) => to.id !== from.id)
      .map((to) => ({
        from: from.id,
        to: to.id,
        oneWayLatencyMs: 0.1,
        jitterP95Ms: 0,
        bandwidthMbps: 10_000,
        lossRate: 0,
        availability: 0.999,
      })),
  );
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
      if (!address || typeof address === "string") {
        throw new Error("port_reservation_failed");
      }
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
