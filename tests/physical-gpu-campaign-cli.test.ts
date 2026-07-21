import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
import {
  PHYSICAL_GPU_CAMPAIGN_CLI_OBSERVATION_SCHEMA,
  PHYSICAL_GPU_CAMPAIGN_CONFIG_SCHEMA,
  executePhysicalGpuCampaignCli,
  extractPhysicalGpuRankWork,
  parsePhysicalGpuCampaignCliArguments,
  parsePhysicalGpuCampaignCliConfig,
  validateRankWorkAgainstCampaign,
  type PhysicalGpuCampaignCliConfig,
  type PhysicalGpuCampaignCliRemoteAgent,
} from "../src/distribution/physical-gpu-campaign-cli.js";
import {
  PHYSICAL_GPU_CAMPAIGN_SCHEMA,
  type PhysicalGpuCampaignInput,
  type PhysicalGpuCampaignObservation,
} from "../src/distribution/physical-gpu-campaign.js";
import {
  buildPhysicalGpuCampaignGateReport,
  type PhysicalGpuCampaignReportHostBinding,
  type PhysicalGpuCampaignReportInput,
} from "../src/distribution/physical-gpu-campaign-report.js";
import { outputTokenIdsSha256 } from "../src/distribution/physical-gpu-gate-report.js";
import {
  compilePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "../src/distribution/python-launcher.js";
import type { PhysicalProbeV1 } from "../src/distribution/physical-probe.js";
import {
  buildRuntimePipelineManifest,
  type RuntimeNodeProfile,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
} from "../src/distribution/types.js";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const TOKENS = Array.from({ length: 16 }, (_, index) => 400 + index);
const TOKEN_DIGEST = outputTokenIdsSha256(TOKENS);
const SECRET_ROOT = "root-agent-secret-01234567890123456789012";
const SECRET_FINAL = "final-agent-secret-0123456789012345678901";
const SECRET_A = "agent-a-secret-012345678901234567890123";
const SECRET_B = "agent-b-secret-012345678901234567890123";

describe("physical GPU campaign CLI", () => {
  it("parses only the four explicit file flags", () => {
    expect(
      parsePhysicalGpuCampaignCliArguments([
        "--launch",
        "launch.json",
        "--config",
        "campaign.json",
        "--observation-out",
        "observation.json",
        "--report-out",
        "report.json",
      ]),
    ).toEqual({
      launchPath: "launch.json",
      configPath: "campaign.json",
      observationOutPath: "observation.json",
      reportOutPath: "report.json",
    });
    expect(() =>
      parsePhysicalGpuCampaignCliArguments([
        "--launch",
        "launch.json",
        "--auth-token",
        SECRET_A,
      ]),
    ).toThrow("physical_gpu_campaign_cli_argument_is_invalid:--auth-token");
    expect(() =>
      parsePhysicalGpuCampaignCliArguments([
        "--launch",
        "one.json",
        "--launch",
        "two.json",
      ]),
    ).toThrow("physical_gpu_campaign_cli_argument_is_invalid:--launch");
  });

  it("closed-validates config, rejects loopback and never accepts an inline token", () => {
    const config = campaignConfig();
    expect(parsePhysicalGpuCampaignCliConfig(config)).toEqual(config);

    const withSecret = structuredClone(config) as unknown as Record<string, unknown>;
    (withSecret.hosts as Array<Record<string, unknown>>)[0]!.authToken = SECRET_A;
    expect(() => parsePhysicalGpuCampaignCliConfig(withSecret)).toThrow(
      /^physical_gpu_campaign_cli_config_is_invalid:/,
    );

    const loopback = structuredClone(config);
    loopback.agents[0]!.endpoint = "http://127.0.0.1:9750";
    expect(() => parsePhysicalGpuCampaignCliConfig(loopback)).toThrow(
      "physical_gpu_campaign_cli_agent_endpoint_is_invalid",
    );
  });

  it("never overwrites an input or a pre-existing evidence artifact", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gdlp-physical-cli-immutable-"));
    try {
      const launch = launchDescription();
      await writeInputs(directory, launch, campaignConfig());
      const launchText = await readFile(join(directory, "launch.json"), "utf8");
      let stderr = "";
      const collidingExit = await executePhysicalGpuCampaignCli(
        [
          "--launch",
          "launch.json",
          "--config",
          "config.json",
          "--observation-out",
          "launch.json",
          "--report-out",
          "report.json",
        ],
        {
          cwd: directory,
          writeStderr: (message) => {
            stderr += message;
          },
        },
      );
      expect(collidingExit).toBe(1);
      expect(stderr).toContain("physical_gpu_campaign_cli_all_paths_must_differ");
      expect(await readFile(join(directory, "launch.json"), "utf8")).toBe(launchText);

      await writeFile(join(directory, "observation.json"), "immutable-old-evidence", "utf8");
      const createAgent = vi.fn();
      const existingExit = await executePhysicalGpuCampaignCli(cliArguments(), {
        cwd: directory,
        createAgent,
        writeStderr: () => undefined,
      });
      expect(existingExit).toBe(1);
      expect(createAgent).not.toHaveBeenCalled();
      expect(await readFile(join(directory, "observation.json"), "utf8")).toBe(
        "immutable-old-evidence",
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("probes both physical hosts before launch, writes sealed artifacts, and redacts credentials", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gdlp-physical-cli-success-"));
    try {
      const launch = launchDescription();
      const config = campaignConfig();
      await writeInputs(directory, launch, config);
      const events: string[] = [];
      const created: Array<{ endpoint: string; id: string; authToken: string }> = [];
      let reportInput: PhysicalGpuCampaignReportInput | null = null;
      const campaign = successfulCampaign(launch);

      const exitCode = await executePhysicalGpuCampaignCli(cliArguments(), {
        cwd: directory,
        environment: {
          TOKEN_ROOT: SECRET_ROOT,
          TOKEN_FINAL: SECRET_FINAL,
          TOKEN_A: SECRET_A,
          TOKEN_B: SECRET_B,
        },
        now: () => new Date("2026-07-21T15:00:00.000Z"),
        createAgent: (options) => {
          const id = options.id!;
          const endpoint = String(options.endpoint);
          const authToken = options.authToken!;
          created.push({ endpoint, id, authToken });
          const index = id.endsWith("node-a") ? 0 : 1;
          return fakeAgent(id, async (nonce) => {
            events.push(`probe:${id}:${nonce}`);
            return physicalProbe(index, nonce);
          });
        },
        runCampaign: async (input) => {
          events.push("campaign");
          const campaignInput = input as PhysicalGpuCampaignInput;
          expect(campaignInput.agents).toHaveLength(4);
          expect(campaignInput.agents.map((binding) => binding.agent.id)).toEqual([
            "local-process:root-node",
            "local-process:node-a",
            "local-process:final-node",
            "local-process:node-b",
          ]);
          return campaign;
        },
        buildReport: (input) => {
          reportInput = input as PhysicalGpuCampaignReportInput;
          return buildPhysicalGpuCampaignGateReport(input);
        },
      });

      expect(exitCode).toBe(0);
      expect(events).toEqual([
        "probe:local-process:node-a:probe-node-a-0001",
        "probe:local-process:node-b:probe-node-b-0002",
        "campaign",
      ]);
      expect(created).toEqual([
        {
          endpoint: "http://10.20.0.11:9749",
          id: "local-process:root-node",
          authToken: SECRET_ROOT,
        },
        {
          endpoint: "http://10.20.0.11:9750",
          id: "local-process:node-a",
          authToken: SECRET_A,
        },
        {
          endpoint: "http://10.20.0.11:9751",
          id: "local-process:final-node",
          authToken: SECRET_FINAL,
        },
        {
          endpoint: "http://10.20.0.12:9750",
          id: "local-process:node-b",
          authToken: SECRET_B,
        },
      ]);
      expect(reportInput).not.toBeNull();
      expect(reportInput!.samplesTruncated).toBe(false);
      expect(reportInput!.rankWork).toEqual([
        expect.objectContaining({
          rank: 0,
          hostId: "host-a",
          nodeId: "node-a",
          device: "cuda:0",
          forwardCalls: 500,
          collectiveCalls: 2_500,
          tokensProcessed: 1_000,
          bytesSent: 0,
          bytesReceived: 0,
          peakAllocatedBytes: GIB,
        }),
        expect.objectContaining({
          rank: 1,
          hostId: "host-b",
          nodeId: "node-b",
          device: "cuda:0",
          forwardCalls: 500,
          collectiveCalls: 2_500,
          tokensProcessed: 1_000,
          bytesSent: 0,
          bytesReceived: 0,
          peakAllocatedBytes: GIB,
        }),
      ]);

      const observationText = await readFile(join(directory, "observation.json"), "utf8");
      const reportText = await readFile(join(directory, "report.json"), "utf8");
      const observation = JSON.parse(observationText) as Record<string, unknown>;
      expect(observation.schema).toBe(PHYSICAL_GPU_CAMPAIGN_CLI_OBSERVATION_SCHEMA);
      expect(observation.passed).toBe(true);
      expect(observation.campaign).not.toBeNull();
      expect(observation.failure).toBeNull();
      expect(observation.probes).toHaveLength(2);
      expect(observationText).not.toContain(SECRET_ROOT);
      expect(observationText).not.toContain(SECRET_FINAL);
      expect(observationText).not.toContain(SECRET_A);
      expect(observationText).not.toContain(SECRET_B);
      expect(reportText).not.toContain(SECRET_A);
      expect(reportText).not.toContain(SECRET_B);
      expect((JSON.parse(reportText) as { gate: { passed: boolean } }).gate.passed).toBe(true);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("writes a redacted observation and never launches when a physical probe fails", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gdlp-physical-cli-probe-fail-"));
    try {
      await writeInputs(directory, launchDescription(), campaignConfig());
      const runCampaign = vi.fn();
      const buildReport = vi.fn();
      let stderr = "";

      const exitCode = await executePhysicalGpuCampaignCli(cliArguments(), {
        cwd: directory,
        environment: {
          TOKEN_ROOT: SECRET_ROOT,
          TOKEN_FINAL: SECRET_FINAL,
          TOKEN_A: SECRET_A,
          TOKEN_B: SECRET_B,
        },
        createAgent: (options) => {
          const id = options.id!;
          const index = id.endsWith("node-a") ? 0 : 1;
          return fakeAgent(id, async (nonce) => {
            if (index === 1) throw new Error(`probe leaked ${SECRET_B}`);
            return physicalProbe(index, nonce);
          });
        },
        runCampaign,
        buildReport,
        writeStderr: (message) => {
          stderr += message;
        },
      });

      expect(exitCode).toBe(1);
      expect(runCampaign).not.toHaveBeenCalled();
      expect(buildReport).not.toHaveBeenCalled();
      const text = await readFile(join(directory, "observation.json"), "utf8");
      const observation = JSON.parse(text) as {
        failure: { phase: string; message: string };
        campaign: unknown;
      };
      expect(observation.failure).toEqual({
        phase: "probe",
        message: "probe leaked [REDACTED]",
      });
      expect(observation.campaign).toBeNull();
      expect(text).not.toContain(SECRET_A);
      expect(text).not.toContain(SECRET_B);
      expect(stderr).not.toContain(SECRET_B);
      await expect(readFile(join(directory, "report.json"), "utf8")).rejects.toThrow();
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("preserves the collector result but refuses truncated process evidence", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gdlp-physical-cli-truncated-"));
    try {
      const launch = launchDescription();
      await writeInputs(directory, launch, campaignConfig());
      const campaign = successfulCampaign(launch);
      const anchor = campaign.lifecycle.supervisorStopped!.processes.find(
        (process) => process.kind === "remote-stage" && process.stageIndex === 1,
      )!;
      anchor.output!.stderrTruncated = true;
      const buildReport = vi.fn();

      const exitCode = await executePhysicalGpuCampaignCli(cliArguments(), {
        cwd: directory,
        environment: {
          TOKEN_ROOT: SECRET_ROOT,
          TOKEN_FINAL: SECRET_FINAL,
          TOKEN_A: SECRET_A,
          TOKEN_B: SECRET_B,
        },
        createAgent: (options) => {
          const id = options.id!;
          const index = id.endsWith("node-a") ? 0 : 1;
          return fakeAgent(id, (nonce) => Promise.resolve(physicalProbe(index, nonce)));
        },
        runCampaign: async () => campaign,
        buildReport,
        writeStderr: () => undefined,
      });

      expect(exitCode).toBe(1);
      expect(buildReport).not.toHaveBeenCalled();
      const observation = JSON.parse(
        await readFile(join(directory, "observation.json"), "utf8"),
      ) as { campaign: { passed: boolean }; failure: { phase: string; message: string } };
      expect(observation.campaign.passed).toBe(true);
      expect(observation.failure).toEqual({
        phase: "rank_work",
        message: "physical_gpu_campaign_cli_process_output_is_truncated",
      });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("returns non-zero and preserves the collector observation when the campaign fails", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gdlp-physical-cli-campaign-fail-"));
    try {
      const launch = launchDescription();
      await writeInputs(directory, launch, campaignConfig(launch));
      const campaign = successfulCampaign(launch);
      campaign.passed = false;
      campaign.failures.push({ phase: "benchmark", message: "exact_parity_failed" });
      const buildReport = vi.fn();
      const exitCode = await executePhysicalGpuCampaignCli(cliArguments(), {
        cwd: directory,
        environment: {
          TOKEN_ROOT: SECRET_ROOT,
          TOKEN_FINAL: SECRET_FINAL,
          TOKEN_A: SECRET_A,
          TOKEN_B: SECRET_B,
        },
        createAgent: (options) => {
          const id = options.id!;
          const index = id.endsWith("node-a") ? 0 : 1;
          return fakeAgent(id, (nonce) => Promise.resolve(physicalProbe(index, nonce)));
        },
        runCampaign: async () => campaign,
        buildReport,
        writeStderr: () => undefined,
      });

      expect(exitCode).toBe(1);
      expect(buildReport).not.toHaveBeenCalled();
      const observation = JSON.parse(
        await readFile(join(directory, "observation.json"), "utf8"),
      ) as { campaign: { passed: boolean }; failure: { phase: string; message: string } };
      expect(observation.campaign.passed).toBe(false);
      expect(observation.failure).toEqual({
        phase: "campaign",
        message: "physical_gpu_campaign_cli_campaign_did_not_pass",
      });
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("uses the latest cumulative rank report and rejects a counter decrement", () => {
    const launch = launchDescription();
    const campaign = successfulCampaign(launch);
    const hosts = reportHosts();
    const rankWork = extractPhysicalGpuRankWork(
      launch,
      campaign.lifecycle.supervisorStopped,
      hosts,
    );
    expect(rankWork.map((work) => work.forwardCalls)).toEqual([500, 500]);
    const insufficient = structuredClone(rankWork);
    insufficient[0]!.tokensProcessed = 47;
    expect(() => validateRankWorkAgainstCampaign(insufficient, campaign)).toThrow(
      "physical_gpu_campaign_cli_rank_work_is_insufficient_for_campaign",
    );

    const stopped = structuredClone(campaign.lifecycle.supervisorStopped!);
    const anchor = stopped.processes.find(
      (process) => process.kind === "remote-stage" && process.stageIndex === 1,
    )!;
    anchor.output!.stderr += `${rankMetricLine(20, 100, 40, GIB)}\n`;
    expect(() => extractPhysicalGpuRankWork(launch, stopped, hosts)).toThrow(
      "physical_gpu_campaign_cli_rank_work_is_not_monotonic",
    );
  });
});

function cliArguments(): string[] {
  return [
    "--launch",
    "launch.json",
    "--config",
    "config.json",
    "--observation-out",
    "observation.json",
    "--report-out",
    "report.json",
  ];
}

async function writeInputs(
  directory: string,
  launch: PythonPipelineLaunchDescription,
  config: PhysicalGpuCampaignCliConfig,
): Promise<void> {
  await Promise.all([
    writeFile(join(directory, "launch.json"), JSON.stringify(launch), "utf8"),
    writeFile(join(directory, "config.json"), JSON.stringify(config), "utf8"),
  ]);
}

function campaignConfig(
  launch: PythonPipelineLaunchDescription = launchDescription(),
): PhysicalGpuCampaignCliConfig {
  const canaries = ["ref", "syntax", "context"].map((id) => ({
    id: `canary-${id}`,
    messages: [{ role: "user" as const, content: `case-${id}` }],
    maxTokens: TOKENS.length,
    expectedOutputTokenIdsSha256: TOKEN_DIGEST,
    expectedCompletionTokens: TOKENS.length,
    expectedFinishReason: "length" as const,
  }));
  return {
    schema: PHYSICAL_GPU_CAMPAIGN_CONFIG_SCHEMA,
    networkScope: "lan",
    apiBaseUrl: "http://10.20.0.11:8081",
    agents: [
      {
        nodeId: "root-node",
        endpoint: "http://10.20.0.11:9749",
        authTokenEnv: "TOKEN_ROOT",
      },
      {
        nodeId: "node-a",
        endpoint: "http://10.20.0.11:9750",
        authTokenEnv: "TOKEN_A",
      },
      {
        nodeId: "final-node",
        endpoint: "http://10.20.0.11:9751",
        authTokenEnv: "TOKEN_FINAL",
      },
      {
        nodeId: "node-b",
        endpoint: "http://10.20.0.12:9750",
        authTokenEnv: "TOKEN_B",
      },
    ],
    hosts: [
      {
        hostId: "host-a",
        rankNodeId: "node-a",
        agentNodeId: "node-a",
        device: "cuda:0",
        offeredVramBytes: 2 * GIB,
        vendor: "NVIDIA",
        nonce: "probe-node-a-0001",
      },
      {
        hostId: "host-b",
        rankNodeId: "node-b",
        agentNodeId: "node-b",
        device: "cuda:0",
        offeredVramBytes: 2 * GIB,
        vendor: "NVIDIA",
        nonce: "probe-node-b-0002",
      },
    ],
    canaries,
    warmups: 1,
    iterations: 5,
    concurrencies: [3],
    networkLinks: [
      {
        fromHostId: "host-a",
        toHostId: "host-b",
        rttMs: [1.1, 1.2, 1.15],
        goodputMbps: [900, 910, 905],
      },
      {
        fromHostId: "host-b",
        toHostId: "host-a",
        rttMs: [1.2, 1.25, 1.18],
        goodputMbps: [880, 890, 885],
      },
    ],
    reference: {
      referenceCanaryId: "canary-ref",
      artifactIdentity: launch.runtimeModel.artifactIdentity!,
      canonicalSource: launch.runtimeModel.canonicalSource!,
      canonicalRevision: launch.runtimeModel.canonicalRevision ?? null,
      tokenizerId: launch.modelIdentity.tokenizerId,
      promptTokenIdsSha256: `sha256:${"f".repeat(64)}`,
      outputTokenIds: [...TOKENS],
    },
  };
}

function fakeAgent(
  id: string,
  physicalEvidence: (nonce: string) => Promise<PhysicalProbeV1>,
): PhysicalGpuCampaignCliRemoteAgent {
  return {
    id,
    async start() {
      throw new Error("campaign runner was injected");
    },
    async health() {
      return {
        schema: "gdlp-launch-agent-health/2",
        agentId: id,
        nodeId: id.slice("local-process:".length),
        activeProcesses: 0,
        retainedTombstones: 0,
      };
    },
    physicalEvidence,
  };
}

function physicalProbe(index: number, nonce: string): PhysicalProbeV1 {
  const suffix = index === 0 ? "a" : "b";
  return {
    schema: "gdlp-physical-probe/1",
    nonce,
    host: {
      fingerprintSha256: `sha256:${suffix.repeat(64)}`,
      fingerprintSource: "machine-id",
      platform: "linux",
      architecture: "x86_64",
      kernelRelease: "6.8.0",
      pythonVersion: "3.12.4",
    },
    runtime: {
      torchVersion: "2.7.0",
      cudaVersion: "12.8",
      rocmVersion: null,
      cudaApiAvailable: true,
      distributedAvailable: true,
      ncclAvailable: true,
      ncclVersion: "2.26.2",
    },
    devices: [
      {
        index: 0,
        name: `Domestic GPU ${index}`,
        totalMemoryBytes: 4 * GIB,
        freeMemoryBytes: 3 * GIB,
        runtimeTotalMemoryBytes: 4 * GIB,
        capability: [8, 6],
        uuidSha256: `sha256:${String(index + 1).repeat(64)}`,
        fingerprintSha256: `sha256:${String(index + 3).repeat(64)}`,
      },
    ],
  };
}

function successfulCampaign(
  launch: PythonPipelineLaunchDescription,
): PhysicalGpuCampaignObservation {
  const processBase = launch.launchOrder.map((process) => ({
    processId: process.processId,
    nodeId: process.anchor.memberId,
    kind: process.kind,
    stageIndex: process.stageIndex,
  }));
  const running = {
    launchId: launch.launchId,
    pipelineId: launch.pipelineId,
    state: "running" as const,
    failure: null,
    processes: processBase.map((process) => ({ ...process, state: "ready" as const })),
    telemetry: [],
    telemetryDropped: 0,
  };
  const stopped = {
    launchId: launch.launchId,
    pipelineId: launch.pipelineId,
    state: "stopped" as const,
    failure: null,
    processes: processBase.map((process) => ({
      ...process,
      state: "stopped" as const,
      output: {
        stdout: "",
        stderr:
          process.kind === "remote-stage" && process.stageIndex === 1
            ? `${rankMetricLine(250, 1_250, 500, 768 * MIB)}\n${rankMetricLine(500, 2_500, 1_000, GIB)}\n`
            : "",
        stdoutTruncated: false,
        stderrTruncated: false,
      },
    })),
    telemetry: [],
    telemetryDropped: 0,
  };
  const emptyStats = {
    count: 15,
    mean: 50,
    p50: 50,
    p95: 50,
    min: 50,
    max: 50,
  };
  const canaryEvidence = {
    promptTokens: 12,
    completionTokens: 16,
    finishReason: "length" as const,
    outputTokenIdsSha256: TOKEN_DIGEST,
    outputTokenIdsHashScheme: "gdlp-output-token-ids-v1" as const,
    serverTtftMs: 10,
    serverTpotMs: 2,
    serverPipelineMs: 40,
  };
  const canaryIds = ["canary-ref", "canary-syntax", "canary-context"];
  const canaryObservations = (phase: "pre" | "post") =>
    canaryIds.map((canaryId) => ({
      phase,
      canaryId,
      passed: true,
      clientResponseMs: 50,
      evidence: { ...canaryEvidence },
      error: null,
    }));
  const sample = (
    phase: "warmup" | "measure",
    iteration: number,
    requestIndex: number,
  ) => ({
    sampleId: `${phase}-c3-i${iteration}-r${requestIndex}`,
    phase,
    canaryId: canaryIds[requestIndex]!,
    concurrency: 3,
    iteration,
    requestIndex,
    passed: true,
    clientFirstContentMs: 12,
    clientResponseMs: 50,
    promptTokens: 12,
    completionTokens: 16,
    finishReason: "length" as const,
    outputTokenIdsSha256: TOKEN_DIGEST,
    serverTtftMs: 10,
    serverTpotMs: 2,
    serverPipelineMs: 40,
    perUserOutputTokensPerSecondIncludingTtft: 320,
    error: null,
  });
  const warmupSamples = [0, 1, 2].map((requestIndex) =>
    sample("warmup", 0, requestIndex),
  );
  const measuredSamples = Array.from({ length: 5 }, (_, iteration) =>
    [0, 1, 2].map((requestIndex) => sample("measure", iteration, requestIndex)),
  ).flat();
  const batches = [
    {
      batchId: "warmup-c3-i0",
      phase: "warmup" as const,
      concurrency: 3,
      iteration: 0,
      passed: true,
      clientWallMs: 50,
      actualCompletionTokens: 48,
      aggregateOutputTokensPerSecondIncludingTtft: 960,
    },
    ...Array.from({ length: 5 }, (_, iteration) => ({
      batchId: `measure-c3-i${iteration}`,
      phase: "measure" as const,
      concurrency: 3,
      iteration,
      passed: true,
      clientWallMs: 50,
      actualCompletionTokens: 48,
      aggregateOutputTokensPerSecondIncludingTtft: 960,
    })),
  ];
  const expectedNodes = [
    ...new Set(launch.launchOrder.map((process) => process.anchor.memberId)),
  ];
  const agentHealth = (phase: "before" | "after") =>
    expectedNodes.map((nodeId) => ({
      phase,
      expectedAgentId: `local-process:${nodeId}`,
      expectedNodeId: nodeId,
      health: {
        schema: "gdlp-launch-agent-health/2" as const,
        agentId: `local-process:${nodeId}`,
        nodeId,
        activeProcesses: 0,
        retainedTombstones: 0,
      },
      passed: true,
      error: null,
    }));
  const root = launch.launchOrder.find((process) => process.kind === "root-engine");
  if (root?.kind !== "root-engine") throw new Error("test_root_missing");
  return {
    schema: PHYSICAL_GPU_CAMPAIGN_SCHEMA,
    launchId: launch.launchId,
    pipelineId: launch.pipelineId,
    apiBaseUrl: "http://10.20.0.11:8081",
    passed: true,
    lifecycle: {
      events: [],
      agentHealthBefore: agentHealth("before"),
      agentHealthAfter: agentHealth("after"),
      supervisorStarted: running,
      supervisorBeforeStop: running,
      supervisorStopped: stopped,
      cleanupAttempted: true,
      cleanupPassed: true,
    },
    apiHealth: {
      status: "ready",
      error: null,
      model: launch.configuration.publicModelName,
      artifactIdentity: launch.runtimeModel.artifactIdentity!,
      canonicalModelSource: launch.runtimeModel.canonicalSource!,
      canonicalModelRevision: launch.runtimeModel.canonicalRevision ?? null,
      pipelineSnapshotIdentity: launch.runtimeModel.snapshotIdentity!,
      stages: launch.route.logicalStageIds.length,
      boundaries: [...root.boundaries],
      codec: launch.route.codec,
    },
    canaries: {
      pre: canaryObservations("pre"),
      post: canaryObservations("post"),
    },
    samples: [...warmupSamples, ...measuredSamples],
    batches,
    summary: {
      measuredRequests: 15,
      actualCompletionTokens: 240,
      measuredBatchWallMs: 250,
      aggregateOutputTokensPerSecondIncludingTtft: 960,
      clientFirstContentMs: emptyStats,
      clientResponseMs: emptyStats,
      serverTtftMs: emptyStats,
      serverTpotMs: emptyStats,
      serverPipelineMs: emptyStats,
      byConcurrency: [
        {
          concurrency: 3,
          measuredRequests: 15,
          actualCompletionTokens: 240,
          measuredBatchWallMs: 250,
          aggregateOutputTokensPerSecondIncludingTtft: 960,
        },
      ],
    },
    failures: [],
  };
}

function rankMetricLine(
  forwardCalls: number,
  collectiveCalls: number,
  tokensProcessed: number,
  peakAllocatedBytes: number,
): string {
  return JSON.stringify({
    request_id: forwardCalls,
    bytes_out: 1_024,
    cell_rank_work: [0, 1].map((rank) => ({
      rank,
      device: "cuda:0",
      forwardCalls,
      collectiveCalls,
      tokensProcessed,
      memory: {
        allocatedBytes: Math.floor(peakAllocatedBytes / 2),
        reservedBytes: peakAllocatedBytes,
        peakAllocatedBytes,
      },
    })),
  });
}

function reportHosts(): PhysicalGpuCampaignReportHostBinding[] {
  const config = campaignConfig();
  return config.hosts.map((host, index) => ({
    hostId: host.hostId,
    agentId: `local-process:${host.rankNodeId}`,
    agentEndpoint: config.agents.find((agent) => agent.nodeId === host.agentNodeId)!.endpoint,
    rankNodeId: host.rankNodeId,
    device: host.device,
    offeredVramBytes: host.offeredVramBytes,
    vendor: host.vendor,
    expectedProbeNonce: host.nonce,
    probe: physicalProbe(index, host.nonce),
  }));
}

function launchDescription(): PythonPipelineLaunchDescription {
  const nodes = [
    pipelineNode("root-node", "10.20.0.11", 22_000),
    cellNode("node-a", "10.20.0.11", 22_100),
    pipelineNode("final-node", "10.20.0.11", 22_150),
    cellNode("node-b", "10.20.0.12", 22_200),
  ];
  const plan: DistributionPlan = {
    algorithm: "physical-cli-test",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: [
      { nodeId: "root-node", layerStart: 0, layerEnd: 2 },
      { nodeId: "node-a", layerStart: 2, layerEnd: 4 },
      { nodeId: "final-node", layerStart: 4, layerEnd: 6 },
    ],
  };
  const request: RuntimePlanRequest = {
    model: modelProfile(),
    modelRevision: `sha256:${"d".repeat(64)}`,
    tokenizerId: "physical-cli-tokenizer",
    topology: { nodes, links: completeLinks(nodes) },
    workload: {
      promptTokens: 12,
      outputTokens: 16,
      contextTokens: 64,
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
        memberNodeIds: ["node-a", "node-b"],
        execution: {
          mode: "tensor-parallel-cell",
          engine: "python-torch",
          collectiveBackend: "nccl",
          computeDtype: "float16",
          fixture: {
            schema: "gdlp-llama-cell-stage/2",
            location: "member-local",
            path: "/srv/gdlp/cell-rank-0",
            layerCount: 2,
            manifestSha256: "1".repeat(64),
            shardSha256: ["2".repeat(64), "3".repeat(64)],
            rankMemory: [0, 1].map(() => ({
              fixedBytes: 8 * MIB,
              kvBytesPerToken: 1_024,
              requiredBytes: 8 * MIB + 1_024 * 64,
            })),
          },
          worldSize: 2,
          rankMemberIds: ["node-a", "node-b"],
          rankWeights: [1, 1],
          rankDevices: ["cuda:0", "cuda:0"],
          operationTimeoutSeconds: 60,
          external: {
            rankFixturePaths: [
              "/srv/gdlp/cell-rank-0",
              "/srv/gdlp/cell-rank-1",
            ],
            controlBindHost: "0.0.0.0",
            controlAdvertiseHost: "10.20.0.11",
            controlPort: 29_100,
            distributedAdvertiseHost: "10.20.0.11",
            distributedPort: 29_101,
            startupTimeoutSeconds: 60,
          },
        },
      },
    ],
  };
  return compilePythonLaunchDescription(buildRuntimePipelineManifest(request), {
    apiEndpoint: { host: "0.0.0.0", port: 8_081 },
    returnEndpoint: { host: "10.20.0.11", port: 30_000 },
    returnBindHost: "0.0.0.0",
    runtimeModel: {
      source: "example/physical-cli-model",
      revision: "d".repeat(40),
    },
    publicModelName: "physical-cli-model",
    maxOutputTokens: 32,
  });
}

function modelProfile(): DistributedModelProfile {
  return {
    id: "physical-cli-model",
    layers: Array.from({ length: 6 }, (_, index) => ({
      index,
      weightBytes: 8 * MIB,
      activationElements: 512,
      kvBytesPerToken: 128,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: MIB,
    lmHeadBytes: MIB,
    runtimeOverheadBytesPerStage: MIB,
    embeddingDecodeMsAtUnit: 0.1,
    lmHeadDecodeMsAtUnit: 0.1,
    embeddingPrefillMsPerTokenAtUnit: 0.01,
    lmHeadPrefillMsPerTokenAtUnit: 0.01,
  };
}

function pipelineNode(id: string, host: string, port: number): RuntimeNodeProfile {
  return {
    ...baseNode(id, host, port),
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

function cellNode(id: string, host: string, port: number): RuntimeNodeProfile {
  return {
    ...baseNode(id, host, port),
    backend: {
      engine: "python-torch",
      version: "2",
      modelFormats: ["safetensors"],
      executionModes: ["tensor-parallel-cell"],
    },
    capabilities: {
      deviceKinds: ["gpu"],
      computeApis: ["nccl", "cuda"],
      weightDtypes: ["fp16"],
      activationCodecs: ["fp16"],
      features: ["rank-local-kv"],
    },
  };
}

function baseNode(id: string, host: string, port: number) {
  return {
    id,
    region: "physical-cli-test",
    memoryBytes: 128 * MIB,
    reserveBytes: 8 * MIB,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0,
    maxBatchSpeedup: 1,
    powerWatts: 75,
    availability: 0.999,
    endpoint: { host, port },
  };
}

function completeLinks(nodes: RuntimeNodeProfile[]) {
  return nodes.flatMap((from) =>
    nodes
      .filter((to) => to.id !== from.id)
      .map((to) => ({
        from: from.id,
        to: to.id,
        oneWayLatencyMs: 0.6,
        jitterP95Ms: 0.1,
        bandwidthMbps: 1_000,
        lossRate: 0,
        availability: 0.999,
      })),
  );
}
