import { randomUUID } from "node:crypto";
import {
  access,
  mkdir,
  link,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../core/json.js";
import {
  HttpLaunchAgent,
  type HttpLaunchAgentOptions,
} from "./launch-agent-rpc.js";
import type {
  LaunchAgent,
  LaunchSupervisorSnapshot,
} from "./launch-supervisor.js";
import {
  runPhysicalGpuCampaign,
  type PhysicalGpuCampaignCanary,
  type PhysicalGpuCampaignObservation,
} from "./physical-gpu-campaign.js";
import {
  buildPhysicalGpuCampaignGateReport,
  type PhysicalGpuCampaignReportHostBinding,
  type PhysicalGpuCampaignReportInput,
  type PhysicalGpuCampaignReportReference,
} from "./physical-gpu-campaign-report.js";
import {
  outputTokenIdsSha256,
  type PhysicalGateDirectionalLinkEvidenceV1,
  type PhysicalGateRankWorkEvidenceV1,
  type PhysicalTwoHostGpuGateReportV1,
} from "./physical-gpu-gate-report.js";
import {
  validatePhysicalProbe,
  type PhysicalProbeV1,
} from "./physical-probe.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
  type PythonRemoteStageLaunch,
} from "./python-launcher.js";

export const PHYSICAL_GPU_CAMPAIGN_CONFIG_SCHEMA =
  "gdlp-physical-gpu-campaign-config/1" as const;
export const PHYSICAL_GPU_CAMPAIGN_CLI_OBSERVATION_SCHEMA =
  "gdlp-physical-gpu-campaign-cli-observation/1" as const;

const IDENTIFIER_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$/;
const ENVIRONMENT_NAME_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NONCE_PATTERN = /^[A-Za-z0-9._:-]{16,128}$/;
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const CUDA_DEVICE_PATTERN = /^cuda:(?:0|[1-9][0-9]*)$/;
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;

export interface PhysicalGpuCampaignCliArguments {
  launchPath: string;
  configPath: string;
  observationOutPath: string;
  reportOutPath: string;
}

export interface PhysicalGpuCampaignCliHostConfig {
  hostId: string;
  rankNodeId: string;
  agentNodeId: string;
  device: string;
  offeredVramBytes: number;
  vendor: string;
  nonce: string;
}

export interface PhysicalGpuCampaignCliAgentConfig {
  nodeId: string;
  endpoint: string;
  authTokenEnv: string;
}

export interface PhysicalGpuCampaignCliConfig {
  schema: typeof PHYSICAL_GPU_CAMPAIGN_CONFIG_SCHEMA;
  networkScope: "lan" | "wan";
  apiBaseUrl: string;
  agents: PhysicalGpuCampaignCliAgentConfig[];
  hosts: [PhysicalGpuCampaignCliHostConfig, PhysicalGpuCampaignCliHostConfig];
  canaries: PhysicalGpuCampaignCanary[];
  warmups: number;
  iterations: number;
  concurrencies: number[];
  networkLinks: PhysicalGateDirectionalLinkEvidenceV1[];
  reference: PhysicalGpuCampaignReportReference;
}

export interface PhysicalGpuCampaignCliProbeObservation {
  hostId: string;
  rankNodeId: string;
  agentId: string;
  agentEndpoint: string;
  nonce: string;
  probe: PhysicalProbeV1;
}

export type PhysicalGpuCampaignCliFailurePhase =
  | "input"
  | "probe"
  | "campaign"
  | "rank_work"
  | "gate"
  | "report_write";

export interface PhysicalGpuCampaignCliFailure {
  phase: PhysicalGpuCampaignCliFailurePhase;
  message: string;
}

export interface PhysicalGpuCampaignCliObservationV1 {
  schema: typeof PHYSICAL_GPU_CAMPAIGN_CLI_OBSERVATION_SCHEMA;
  capturedAt: string;
  passed: boolean;
  launchId: string | null;
  pipelineId: string | null;
  launchCanonicalSha256: string | null;
  configCanonicalSha256: string | null;
  probes: PhysicalGpuCampaignCliProbeObservation[];
  campaign: PhysicalGpuCampaignObservation | null;
  reportSealSha256: string | null;
  failure: PhysicalGpuCampaignCliFailure | null;
}

export interface PhysicalGpuCampaignCliRemoteAgent extends LaunchAgent {
  health(signal?: AbortSignal): Promise<unknown>;
  physicalEvidence(nonce: string, signal?: AbortSignal): Promise<PhysicalProbeV1>;
}

export interface PhysicalGpuCampaignCliDependencies {
  cwd?: string;
  environment?: NodeJS.ProcessEnv;
  now?: () => Date;
  readText?: (absolutePath: string) => Promise<string>;
  pathExists?: (absolutePath: string) => Promise<boolean>;
  writeJsonAtomic?: (absolutePath: string, value: unknown) => Promise<void>;
  createAgent?: (
    options: HttpLaunchAgentOptions,
  ) => PhysicalGpuCampaignCliRemoteAgent;
  runCampaign?: typeof runPhysicalGpuCampaign;
  buildReport?: typeof buildPhysicalGpuCampaignGateReport;
  writeStderr?: (message: string) => void;
}

const finitePositive = z.number().positive().finite();
const safePositiveInteger = z.number().int().min(1).max(MAX_SAFE_INTEGER);
const identifier = z.string().regex(IDENTIFIER_PATTERN);
const normalizedText = z
  .string()
  .min(1)
  .max(4_096)
  .refine((value) => !/[\0\r\n]/.test(value));
const sha256 = z.string().regex(SHA256_PATTERN);

const messageSchema = z
  .object({
    role: z.enum(["system", "developer", "user", "assistant"]),
    content: z.string().min(1).max(1_000_000),
  })
  .strict();

const canarySchema = z
  .object({
    id: identifier,
    messages: z.array(messageSchema).min(1).max(1_024),
    maxTokens: safePositiveInteger.max(32_768),
    expectedOutputTokenIdsSha256: sha256,
    expectedCompletionTokens: safePositiveInteger.max(32_768),
    expectedFinishReason: z.enum(["length", "stop"]),
  })
  .strict();

const hostSchema = z
  .object({
    hostId: identifier,
    rankNodeId: identifier,
    agentNodeId: identifier,
    device: z.string().regex(CUDA_DEVICE_PATTERN),
    offeredVramBytes: safePositiveInteger,
    vendor: normalizedText,
    nonce: z.string().regex(NONCE_PATTERN),
  })
  .strict();

const agentSchema = z
  .object({
    nodeId: identifier,
    endpoint: normalizedText,
    authTokenEnv: z.string().regex(ENVIRONMENT_NAME_PATTERN),
  })
  .strict();

const directionalLinkSchema = z
  .object({
    fromHostId: identifier,
    toHostId: identifier,
    rttMs: z.array(finitePositive).min(3).max(100_000),
    goodputMbps: z.array(finitePositive).min(3).max(100_000),
  })
  .strict();

const referenceSchema = z
  .object({
    referenceCanaryId: identifier,
    artifactIdentity: sha256,
    canonicalSource: normalizedText,
    canonicalRevision: normalizedText.nullable(),
    tokenizerId: normalizedText,
    promptTokenIdsSha256: sha256,
    outputTokenIds: z
      .array(z.number().int().min(0).max(0xffff_ffff))
      .min(1)
      .max(32_768),
  })
  .strict();

const configSchema = z
  .object({
    schema: z.literal(PHYSICAL_GPU_CAMPAIGN_CONFIG_SCHEMA),
    networkScope: z.enum(["lan", "wan"]),
    apiBaseUrl: normalizedText,
    agents: z.array(agentSchema).min(3).max(64),
    hosts: z.tuple([hostSchema, hostSchema]),
    canaries: z.array(canarySchema).min(3).max(256),
    warmups: safePositiveInteger.max(1_000),
    iterations: safePositiveInteger.min(5).max(10_000),
    concurrencies: z.array(safePositiveInteger.max(1_024)).min(1).max(32),
    networkLinks: z.tuple([directionalLinkSchema, directionalLinkSchema]),
    reference: referenceSchema,
  })
  .strict();

/** Parse only the four explicit file flags. No secret value is accepted on argv. */
export function parsePhysicalGpuCampaignCliArguments(
  argv: string[],
): PhysicalGpuCampaignCliArguments {
  const allowed = new Set([
    "--launch",
    "--config",
    "--observation-out",
    "--report-out",
  ]);
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (
      name === undefined ||
      !allowed.has(name) ||
      value === undefined ||
      value.length === 0 ||
      /[\0\r\n]/.test(value) ||
      values.has(name)
    ) {
      throw new Error(`physical_gpu_campaign_cli_argument_is_invalid:${name ?? "missing"}`);
    }
    values.set(name, value);
  }
  const launchPath = values.get("--launch");
  const configPath = values.get("--config");
  const observationOutPath = values.get("--observation-out");
  const reportOutPath = values.get("--report-out");
  if (launchPath === undefined) throw new Error("physical_gpu_campaign_cli_launch_is_required");
  if (configPath === undefined) throw new Error("physical_gpu_campaign_cli_config_is_required");
  if (observationOutPath === undefined) {
    throw new Error("physical_gpu_campaign_cli_observation_out_is_required");
  }
  if (reportOutPath === undefined) {
    throw new Error("physical_gpu_campaign_cli_report_out_is_required");
  }
  return { launchPath, configPath, observationOutPath, reportOutPath };
}

export function parsePhysicalGpuCampaignCliConfig(
  value: unknown,
): PhysicalGpuCampaignCliConfig {
  canonicalEvidenceJson(value);
  const parsed = configSchema.safeParse(value);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    const path = issue?.path.length ? issue.path.join(".") : "config";
    throw new Error(
      `physical_gpu_campaign_cli_config_is_invalid:${path}:${issue?.message ?? "unknown"}`,
    );
  }
  const config = structuredClone(parsed.data) as PhysicalGpuCampaignCliConfig;
  config.apiBaseUrl = normalizeRemoteHttpOrigin(config.apiBaseUrl, "api_base_url");
  for (const agent of config.agents) {
    agent.endpoint = normalizeRemoteHttpOrigin(agent.endpoint, "agent_endpoint");
  }
  requireUnique(config.agents.map((agent) => agent.nodeId), "agent_node_id");
  requireUnique(config.agents.map((agent) => agent.endpoint), "agent_endpoint");
  requireUnique(config.hosts.map((host) => host.hostId), "host_id");
  requireUnique(config.hosts.map((host) => host.rankNodeId), "rank_node_id");
  requireUnique(config.hosts.map((host) => host.agentNodeId), "host_agent_node_id");
  requireUnique(config.hosts.map((host) => host.nonce), "probe_nonce");
  requireUnique(config.canaries.map((canary) => canary.id), "canary_id");
  requireUnique(config.concurrencies, "concurrency");
  validateCanaryReference(config);
  validateDirectionalLinks(config.networkLinks, config.hosts);
  return config;
}

/**
 * Run the physical campaign. The numeric result is suitable for process.exitCode.
 * Every failure after the output flags are known is captured in observation-out.
 */
export async function executePhysicalGpuCampaignCli(
  argv: string[],
  dependencies: PhysicalGpuCampaignCliDependencies = {},
): Promise<number> {
  const cwd = resolve(dependencies.cwd ?? process.cwd());
  const environment = dependencies.environment ?? process.env;
  const now = dependencies.now ?? (() => new Date());
  const readText = dependencies.readText ?? ((path) => readFile(path, "utf8"));
  const pathExists = dependencies.pathExists ?? fileExists;
  const writeJsonAtomic = dependencies.writeJsonAtomic ?? atomicWriteJson;
  const createAgent =
    dependencies.createAgent ??
    ((options: HttpLaunchAgentOptions) =>
      new HttpLaunchAgent(options) as PhysicalGpuCampaignCliRemoteAgent);
  const runCampaign = dependencies.runCampaign ?? runPhysicalGpuCampaign;
  const buildReport = dependencies.buildReport ?? buildPhysicalGpuCampaignGateReport;
  const writeStderr = dependencies.writeStderr ?? ((message) => process.stderr.write(message));

  let argumentsValue: PhysicalGpuCampaignCliArguments;
  try {
    argumentsValue = parsePhysicalGpuCampaignCliArguments(argv);
  } catch (error) {
    writeStderr(`${normalizeError(error).message}\n`);
    return 1;
  }
  const observationPath = resolve(cwd, argumentsValue.observationOutPath);
  const reportPath = resolve(cwd, argumentsValue.reportOutPath);
  const launchPath = resolve(cwd, argumentsValue.launchPath);
  const configPath = resolve(cwd, argumentsValue.configPath);
  if (new Set([launchPath, configPath, observationPath, reportPath]).size !== 4) {
    writeStderr("physical_gpu_campaign_cli_all_paths_must_differ\n");
    return 1;
  }
  if (
    (await pathExists(observationPath)) ||
    (await pathExists(reportPath))
  ) {
    writeStderr("physical_gpu_campaign_cli_output_path_already_exists\n");
    return 1;
  }

  const observation: PhysicalGpuCampaignCliObservationV1 = {
    schema: PHYSICAL_GPU_CAMPAIGN_CLI_OBSERVATION_SCHEMA,
    capturedAt: validTimestamp(now()),
    passed: false,
    launchId: null,
    pipelineId: null,
    launchCanonicalSha256: null,
    configCanonicalSha256: null,
    probes: [],
    campaign: null,
    reportSealSha256: null,
    failure: null,
  };
  const resolvedSecrets: string[] = [];
  let phase: PhysicalGpuCampaignCliFailurePhase = "input";
  let report: PhysicalTwoHostGpuGateReportV1 | null = null;

  try {
    const [launchSource, configSource] = await Promise.all([
      readText(launchPath),
      readText(configPath),
    ]);
    const launchValue = parseJson(launchSource, "launch");
    validatePythonLaunchDescription(launchValue);
    const launch = structuredClone(launchValue);
    const config = parsePhysicalGpuCampaignCliConfig(
      parseJson(configSource, "config"),
    );
    validateConfigAgainstLaunch(config, launch);
    observation.launchId = launch.launchId;
    observation.pipelineId = launch.pipelineId;
    observation.launchCanonicalSha256 = sha256CanonicalEvidence(launch);
    observation.configCanonicalSha256 = sha256CanonicalEvidence(config);

    const agents = config.agents.map((agentConfig) => {
      const authToken = environment[agentConfig.authTokenEnv];
      if (authToken === undefined || authToken.length === 0) {
        throw new Error(
          `physical_gpu_campaign_cli_auth_token_env_is_missing:${agentConfig.authTokenEnv}`,
        );
      }
      resolvedSecrets.push(authToken);
      return createAgent({
        endpoint: agentConfig.endpoint,
        id: `local-process:${agentConfig.nodeId}`,
        authToken,
      });
    });
    const agentByNode = new Map(
      config.agents.map((agentConfig, index) => [
        agentConfig.nodeId,
        { config: agentConfig, agent: agents[index]! },
      ]),
    );

    phase = "probe";
    const probes = await Promise.all(
      config.hosts.map(async (host) => {
        const binding = agentByNode.get(host.agentNodeId);
        if (binding === undefined) {
          throw new Error("physical_gpu_campaign_cli_host_agent_is_missing");
        }
        const probe = await binding.agent.physicalEvidence(host.nonce);
        validatePhysicalProbe(probe, host.nonce);
        return probe;
      }),
    );
    validatePhysicalPreflight(config, launch, probes);
    observation.probes = config.hosts.map((host, index) => ({
      hostId: host.hostId,
      rankNodeId: host.rankNodeId,
      agentId: agentByNode.get(host.agentNodeId)!.agent.id,
      agentEndpoint: agentByNode.get(host.agentNodeId)!.config.endpoint,
      nonce: host.nonce,
      probe: structuredClone(probes[index]!),
    }));

    phase = "campaign";
    const campaign = await runCampaign({
      launch,
      agents: config.agents.map((agentConfig, index) => ({
        nodeId: agentConfig.nodeId,
        agent: agents[index]!,
      })),
      apiBaseUrl: config.apiBaseUrl,
      canaries: structuredClone(config.canaries),
      warmups: config.warmups,
      iterations: config.iterations,
      concurrencies: [...config.concurrencies],
    });
    observation.campaign = structuredClone(campaign);
    if (!campaign.passed) {
      throw new Error("physical_gpu_campaign_cli_campaign_did_not_pass");
    }

    const hostBindings = config.hosts.map(
      (host, index): PhysicalGpuCampaignReportHostBinding => ({
        hostId: host.hostId,
        agentId: agentByNode.get(host.agentNodeId)!.agent.id,
        agentEndpoint: agentByNode.get(host.agentNodeId)!.config.endpoint,
        rankNodeId: host.rankNodeId,
        device: host.device,
        offeredVramBytes: host.offeredVramBytes,
        vendor: host.vendor,
        expectedProbeNonce: host.nonce,
        probe: structuredClone(probes[index]!),
      }),
    );

    phase = "rank_work";
    const rankWork = extractPhysicalGpuRankWork(
      launch,
      campaign.lifecycle.supervisorStopped,
      hostBindings,
    );
    validateRankWorkAgainstCampaign(rankWork, campaign);

    phase = "gate";
    const reportInput: PhysicalGpuCampaignReportInput = {
      capturedAt: observation.capturedAt,
      networkScope: config.networkScope,
      samplesTruncated: false,
      campaign,
      launch,
      hosts: hostBindings,
      networkLinks: structuredClone(config.networkLinks),
      reference: structuredClone(config.reference),
      rankWork,
    };
    report = buildReport(reportInput);
    if (!report.gate.passed) {
      throw new Error("physical_gpu_campaign_cli_gate_did_not_pass");
    }
    observation.reportSealSha256 = report.seal.digest;
  } catch (error) {
    observation.failure = {
      phase,
      message: redactSecrets(normalizeError(error).message, resolvedSecrets),
    };
  }

  if (observation.failure === null && report !== null) {
    try {
      await writeJsonAtomic(reportPath, report);
    } catch (error) {
      observation.failure = {
        phase: "report_write",
        message: redactSecrets(normalizeError(error).message, resolvedSecrets),
      };
    }
  }
  observation.passed = observation.failure === null && report !== null;

  try {
    await writeJsonAtomic(observationPath, observation);
  } catch (error) {
    writeStderr(
      `${redactSecrets(normalizeError(error).message, resolvedSecrets)}\n`,
    );
    return 1;
  }
  if (observation.failure !== null) {
    writeStderr(`${observation.failure.phase}:${observation.failure.message}\n`);
    return 1;
  }
  return 0;
}

/**
 * Extract cumulative per-rank physical work from the external cell anchor's
 * JSONL stderr. Rank traffic is not present in that measured document, so it
 * remains zero instead of being estimated or projected.
 */
export function extractPhysicalGpuRankWork(
  launch: PythonPipelineLaunchDescription,
  stoppedValue: LaunchSupervisorSnapshot | null,
  hosts: PhysicalGpuCampaignReportHostBinding[],
): PhysicalGateRankWorkEvidenceV1[] {
  if (stoppedValue === null || stoppedValue.state !== "stopped") {
    throw new Error("physical_gpu_campaign_cli_stopped_snapshot_is_missing");
  }
  const expectedProcessIds = new Set(
    launch.launchOrder.map((process) => process.processId),
  );
  if (
    stoppedValue.processes.length !== expectedProcessIds.size ||
    stoppedValue.processes.some(
      (process) =>
        !expectedProcessIds.has(process.processId) ||
        process.state !== "stopped" ||
        process.output === undefined,
    )
  ) {
    throw new Error("physical_gpu_campaign_cli_stopped_outputs_are_incomplete");
  }
  if (
    stoppedValue.processes.some(
      (process) =>
        process.output!.stdoutTruncated || process.output!.stderrTruncated,
    )
  ) {
    throw new Error("physical_gpu_campaign_cli_process_output_is_truncated");
  }

  const externalStages = launch.launchOrder.filter(
    (process): process is PythonRemoteStageLaunch =>
      process.kind === "remote-stage" && process.cell?.external !== undefined,
  );
  if (externalStages.length !== 1) {
    throw new Error("physical_gpu_campaign_cli_requires_one_external_cell_stage");
  }
  const stage = externalStages[0]!;
  const cell = stage.cell!;
  const process = stoppedValue.processes.find(
    (candidate) => candidate.processId === stage.processId,
  );
  if (process?.output === undefined) {
    throw new Error("physical_gpu_campaign_cli_external_cell_output_is_missing");
  }

  const latest = new Map<number, ParsedCellRankWork>();
  let documents = 0;
  for (const line of process.output.stderr.split(/\r?\n/)) {
    const text = line.trim();
    if (!text.startsWith("{")) continue;
    let value: unknown;
    try {
      value = JSON.parse(text) as unknown;
    } catch {
      throw new Error("physical_gpu_campaign_cli_rank_work_jsonl_is_invalid");
    }
    if (!isRecord(value) || !("cell_rank_work" in value)) continue;
    if (!Array.isArray(value.cell_rank_work) || value.cell_rank_work.length === 0) {
      continue;
    }
    if (value.cell_rank_work.length !== cell.worldSize) {
      throw new Error("physical_gpu_campaign_cli_rank_work_world_size_mismatch");
    }
    const current = value.cell_rank_work.map(parseCellRankWork);
    const ranks = current.map((work) => work.rank);
    if (new Set(ranks).size !== cell.worldSize) {
      throw new Error("physical_gpu_campaign_cli_rank_work_ranks_are_invalid");
    }
    const counterIdentities = new Set(
      current.map(
        (work) =>
          `${work.forwardCalls}\0${work.collectiveCalls}\0${work.tokensProcessed}`,
      ),
    );
    if (counterIdentities.size !== 1) {
      throw new Error("physical_gpu_campaign_cli_rank_work_ranks_disagree");
    }
    for (const work of current) {
      const expectedDevice = cell.rankDevices[work.rank];
      if (expectedDevice === undefined || work.device !== expectedDevice) {
        throw new Error("physical_gpu_campaign_cli_rank_work_device_mismatch");
      }
      const previous = latest.get(work.rank);
      if (previous !== undefined) validateMonotonicRankWork(previous, work);
      latest.set(work.rank, work);
    }
    documents += 1;
  }
  if (documents === 0 || latest.size !== cell.worldSize) {
    throw new Error("physical_gpu_campaign_cli_rank_work_is_missing");
  }

  const hostByNode = new Map(hosts.map((host) => [host.rankNodeId, host]));
  return [...latest.values()]
    .sort((left, right) => left.rank - right.rank)
    .map((work): PhysicalGateRankWorkEvidenceV1 => {
      const nodeId = cell.rankMemberIds[work.rank];
      const host = nodeId === undefined ? undefined : hostByNode.get(nodeId);
      if (nodeId === undefined || host === undefined || host.device !== work.device) {
        throw new Error("physical_gpu_campaign_cli_rank_work_host_mapping_mismatch");
      }
      return {
        rank: work.rank,
        hostId: host.hostId,
        nodeId,
        device: work.device,
        forwardCalls: work.forwardCalls,
        collectiveCalls: work.collectiveCalls,
        tokensProcessed: work.tokensProcessed,
        // The stage JSONL does not expose per-rank wire bytes. Zero means
        // unmeasured here; it must never be replaced with a planner estimate.
        bytesSent: 0,
        bytesReceived: 0,
        peakAllocatedBytes: work.memory.peakAllocatedBytes,
      };
    });
}

export function validateRankWorkAgainstCampaign(
  rankWork: PhysicalGateRankWorkEvidenceV1[],
  campaign: PhysicalGpuCampaignObservation,
): void {
  const canaries = [...campaign.canaries.pre, ...campaign.canaries.post].filter(
    (observation) => observation.passed && observation.evidence !== null,
  );
  const samples = campaign.samples.filter(
    (sample) => sample.passed && sample.completionTokens !== null,
  );
  const minimumRequests = canaries.length + samples.length;
  const minimumTokens =
    canaries.reduce(
      (total, observation) => total + observation.evidence!.completionTokens,
      0,
    ) +
    samples.reduce(
      (total, sample) => total + sample.completionTokens!,
      0,
    );
  if (
    minimumRequests < 1 ||
    minimumTokens < 1 ||
    rankWork.length !== 2 ||
    rankWork.some(
      (work) =>
        work.forwardCalls < minimumRequests || work.tokensProcessed < minimumTokens,
    )
  ) {
    throw new Error("physical_gpu_campaign_cli_rank_work_is_insufficient_for_campaign");
  }
}

interface ParsedCellRankWork {
  rank: number;
  device: string;
  forwardCalls: number;
  collectiveCalls: number;
  tokensProcessed: number;
  memory: {
    allocatedBytes: number;
    reservedBytes: number;
    peakAllocatedBytes: number;
  };
}

function parseCellRankWork(value: unknown): ParsedCellRankWork {
  if (!isRecord(value)) {
    throw new Error("physical_gpu_campaign_cli_rank_work_item_is_invalid");
  }
  exactKeys(
    value,
    [
      "rank",
      "device",
      "forwardCalls",
      "collectiveCalls",
      "tokensProcessed",
      "memory",
    ],
    "physical_gpu_campaign_cli_rank_work_item",
  );
  const rank = safeInteger(value.rank, 0, 255, "rank");
  const device = requiredText(value.device, "device");
  const forwardCalls = safeInteger(value.forwardCalls, 0, MAX_SAFE_INTEGER, "forward_calls");
  const collectiveCalls = safeInteger(
    value.collectiveCalls,
    0,
    MAX_SAFE_INTEGER,
    "collective_calls",
  );
  const tokensProcessed = safeInteger(
    value.tokensProcessed,
    0,
    MAX_SAFE_INTEGER,
    "tokens_processed",
  );
  if (!isRecord(value.memory)) {
    throw new Error("physical_gpu_campaign_cli_rank_work_memory_is_invalid");
  }
  exactKeys(
    value.memory,
    ["allocatedBytes", "reservedBytes", "peakAllocatedBytes"],
    "physical_gpu_campaign_cli_rank_work_memory",
  );
  const allocatedBytes = safeInteger(
    value.memory.allocatedBytes,
    0,
    MAX_SAFE_INTEGER,
    "allocated_bytes",
  );
  const reservedBytes = safeInteger(
    value.memory.reservedBytes,
    0,
    MAX_SAFE_INTEGER,
    "reserved_bytes",
  );
  const peakAllocatedBytes = safeInteger(
    value.memory.peakAllocatedBytes,
    0,
    MAX_SAFE_INTEGER,
    "peak_allocated_bytes",
  );
  if (allocatedBytes > reservedBytes || allocatedBytes > peakAllocatedBytes) {
    throw new Error("physical_gpu_campaign_cli_rank_work_memory_is_inconsistent");
  }
  return {
    rank,
    device,
    forwardCalls,
    collectiveCalls,
    tokensProcessed,
    memory: { allocatedBytes, reservedBytes, peakAllocatedBytes },
  };
}

function validateMonotonicRankWork(
  previous: ParsedCellRankWork,
  current: ParsedCellRankWork,
): void {
  if (previous.device !== current.device) {
    throw new Error("physical_gpu_campaign_cli_rank_work_device_changed");
  }
  const movedBackwards =
    current.forwardCalls < previous.forwardCalls ||
    current.collectiveCalls < previous.collectiveCalls ||
    current.tokensProcessed < previous.tokensProcessed ||
    current.memory.peakAllocatedBytes < previous.memory.peakAllocatedBytes;
  const sameForwardContradiction =
    current.forwardCalls === previous.forwardCalls &&
    (current.collectiveCalls !== previous.collectiveCalls ||
      current.tokensProcessed !== previous.tokensProcessed);
  const newForwardWithoutWork =
    current.forwardCalls > previous.forwardCalls &&
    (current.collectiveCalls <= previous.collectiveCalls ||
      current.tokensProcessed <= previous.tokensProcessed);
  if (movedBackwards || sameForwardContradiction || newForwardWithoutWork) {
    throw new Error("physical_gpu_campaign_cli_rank_work_is_not_monotonic");
  }
}

function validateConfigAgainstLaunch(
  config: PhysicalGpuCampaignCliConfig,
  launch: PythonPipelineLaunchDescription,
): void {
  const launchNodes = [
    ...new Set(launch.launchOrder.map((process) => process.anchor.memberId)),
  ];
  if (
    launchNodes.length < 3 ||
    !sameSet(launchNodes, config.agents.map((agent) => agent.nodeId))
  ) {
    throw new Error("physical_gpu_campaign_cli_agents_do_not_match_launch_nodes");
  }
  const externalStages = launch.launchOrder.filter(
    (process): process is PythonRemoteStageLaunch =>
      process.kind === "remote-stage" && process.cell?.external !== undefined,
  );
  if (externalStages.length !== 1) {
    throw new Error("physical_gpu_campaign_cli_requires_one_external_cell_stage");
  }
  const stage = externalStages[0]!;
  const cell = stage.cell!;
  if (
    cell.fixture.location !== "member-local" ||
    cell.worldSize !== 2 ||
    cell.collectiveBackend !== "nccl" ||
    cell.rankMemberIds.length !== 2 ||
    cell.rankDevices.length !== 2 ||
    !sameSet(cell.rankMemberIds, config.hosts.map((host) => host.rankNodeId))
  ) {
    throw new Error("physical_gpu_campaign_cli_launch_is_not_two_rank_nccl");
  }
  for (let rank = 0; rank < cell.worldSize; rank += 1) {
    const nodeId = cell.rankMemberIds[rank]!;
    const host = config.hosts.find((candidate) => candidate.rankNodeId === nodeId);
    if (
      host === undefined ||
      host.agentNodeId !== nodeId ||
      host.device !== cell.rankDevices[rank] ||
      !config.agents.some((agent) => agent.nodeId === host.agentNodeId)
    ) {
      throw new Error("physical_gpu_campaign_cli_rank_device_binding_mismatch");
    }
  }
  const routeHosts = [
    ...config.agents.map((agent) => new URL(agent.endpoint).hostname),
    new URL(config.apiBaseUrl).hostname,
    cell.external!.controlAdvertiseHost,
    cell.external!.distributedAdvertiseHost,
    ...stage.members.map((member) => member.endpoint.host),
  ];
  if (routeHosts.some((host) => !isRemotelyRoutableHost(host))) {
    throw new Error("physical_gpu_campaign_cli_route_contains_loopback_or_unspecified_host");
  }
  for (const canary of config.canaries) {
    if (canary.maxTokens > launch.configuration.maxOutputTokens) {
      throw new Error("physical_gpu_campaign_cli_canary_exceeds_launch_token_limit");
    }
    if (
      canary.expectedCompletionTokens > canary.maxTokens ||
      (canary.expectedFinishReason === "length" &&
        canary.expectedCompletionTokens !== canary.maxTokens)
    ) {
      throw new Error("physical_gpu_campaign_cli_canary_finish_contract_is_invalid");
    }
  }
  if (
    config.reference.artifactIdentity !== launch.runtimeModel.artifactIdentity ||
    config.reference.canonicalSource !== launch.runtimeModel.canonicalSource ||
    config.reference.canonicalRevision !==
      (launch.runtimeModel.canonicalRevision ?? null) ||
    config.reference.tokenizerId !== launch.modelIdentity.tokenizerId
  ) {
    throw new Error("physical_gpu_campaign_cli_reference_identity_mismatch");
  }
}

function validatePhysicalPreflight(
  config: PhysicalGpuCampaignCliConfig,
  launch: PythonPipelineLaunchDescription,
  probes: PhysicalProbeV1[],
): void {
  if (probes.length !== 2) {
    throw new Error("physical_gpu_campaign_cli_requires_two_physical_probes");
  }
  const hostFingerprints = new Set(
    probes.map((probe) => probe.host.fingerprintSha256),
  );
  if (hostFingerprints.size !== 2) {
    throw new Error("physical_gpu_campaign_cli_probe_hosts_are_not_distinct");
  }
  const deviceFingerprints = new Set<string>();
  for (let index = 0; index < probes.length; index += 1) {
    const probe = probes[index]!;
    const host = config.hosts[index]!;
    validatePhysicalProbe(probe, host.nonce);
    if (
      !probe.runtime.cudaApiAvailable ||
      !probe.runtime.distributedAvailable ||
      !probe.runtime.ncclAvailable
    ) {
      throw new Error("physical_gpu_campaign_cli_probe_is_not_nccl_capable");
    }
    const deviceIndex = Number(host.device.slice("cuda:".length));
    const device = probe.devices.find((candidate) => candidate.index === deviceIndex);
    if (device === undefined) {
      throw new Error("physical_gpu_campaign_cli_probe_device_is_missing");
    }
    if (
      host.offeredVramBytes > device.totalMemoryBytes ||
      host.offeredVramBytes > device.freeMemoryBytes
    ) {
      throw new Error("physical_gpu_campaign_cli_offered_vram_is_not_available");
    }
    deviceFingerprints.add(device.fingerprintSha256);
  }
  if (deviceFingerprints.size !== 2) {
    throw new Error("physical_gpu_campaign_cli_probe_gpus_are_not_distinct");
  }

  // Re-check the binding after probes so a valid GPU cannot mask a route that
  // points the collective rank at another configured host.
  validateConfigAgainstLaunch(config, launch);
}

function validateCanaryReference(config: PhysicalGpuCampaignCliConfig): void {
  const reference = config.reference;
  const matching = config.canaries.filter(
    (canary) => canary.id === reference.referenceCanaryId,
  );
  if (matching.length !== 1) {
    throw new Error("physical_gpu_campaign_cli_reference_canary_is_missing");
  }
  const canary = matching[0]!;
  const digest = outputTokenIdsSha256(reference.outputTokenIds);
  if (
    reference.outputTokenIds.length < 16 ||
    canary.expectedCompletionTokens !== reference.outputTokenIds.length ||
    canary.expectedOutputTokenIdsSha256 !== digest
  ) {
    throw new Error("physical_gpu_campaign_cli_reference_canary_does_not_match_tokens");
  }
  const referenceIndex = config.canaries.findIndex(
    (candidate) => candidate.id === reference.referenceCanaryId,
  );
  const warmupSamples = countCanarySamples(
    referenceIndex,
    config.canaries.length,
    config.warmups,
    config.concurrencies,
  );
  const measuredSamples = countCanarySamples(
    referenceIndex,
    config.canaries.length,
    config.iterations,
    config.concurrencies,
  );
  if (warmupSamples < 1 || measuredSamples < 5) {
    throw new Error("physical_gpu_campaign_cli_reference_sample_count_is_insufficient");
  }
}

function countCanarySamples(
  canaryIndex: number,
  canaryCount: number,
  iterations: number,
  concurrencies: number[],
): number {
  let count = 0;
  for (const concurrency of concurrencies) {
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      for (let requestIndex = 0; requestIndex < concurrency; requestIndex += 1) {
        if ((iteration * concurrency + requestIndex) % canaryCount === canaryIndex) {
          count += 1;
        }
      }
    }
  }
  return count;
}

function validateDirectionalLinks(
  links: PhysicalGateDirectionalLinkEvidenceV1[],
  hosts: readonly PhysicalGpuCampaignCliHostConfig[],
): void {
  const required = new Set([
    `${hosts[0]!.hostId}\0${hosts[1]!.hostId}`,
    `${hosts[1]!.hostId}\0${hosts[0]!.hostId}`,
  ]);
  const observed = new Set<string>();
  for (const link of links) {
    const key = `${link.fromHostId}\0${link.toHostId}`;
    if (!required.has(key) || observed.has(key)) {
      throw new Error("physical_gpu_campaign_cli_network_links_are_not_bidirectional");
    }
    observed.add(key);
  }
  if (observed.size !== required.size) {
    throw new Error("physical_gpu_campaign_cli_network_links_are_not_bidirectional");
  }
}

async function atomicWriteJson(path: string, value: unknown): Promise<void> {
  canonicalEvidenceJson(value);
  const directory = dirname(path);
  await mkdir(directory, { recursive: true });
  const temporary = resolve(
    directory,
    `.${path.split(/[\\/]/).at(-1) ?? "evidence"}.${process.pid}.${randomUUID()}.tmp`,
  );
  try {
    await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
      encoding: "utf8",
      flag: "wx",
    });
    // A hard-link publish is atomic and fails with EEXIST. Physical evidence
    // is append-only: a rerun must use fresh output paths, never replace an
    // older report that an operator may already have inspected.
    await link(temporary, path);
    await rm(temporary, { force: true });
  } catch (error) {
    await rm(temporary, { force: true }).catch(() => undefined);
    throw error;
  }
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT") return false;
    throw error;
  }
}

function parseJson(source: string, kind: "launch" | "config"): unknown {
  try {
    return JSON.parse(source) as unknown;
  } catch {
    throw new Error(`physical_gpu_campaign_cli_${kind}_is_not_json`);
  }
}

function normalizeRemoteHttpOrigin(value: string, name: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`physical_gpu_campaign_cli_${name}_is_invalid`);
  }
  if (
    (url.protocol !== "http:" && url.protocol !== "https:") ||
    url.username !== "" ||
    url.password !== "" ||
    url.search !== "" ||
    url.hash !== "" ||
    (url.pathname !== "" && url.pathname !== "/") ||
    !isRemotelyRoutableHost(url.hostname)
  ) {
    throw new Error(`physical_gpu_campaign_cli_${name}_is_invalid`);
  }
  return url.origin;
}

function isRemotelyRoutableHost(value: string): boolean {
  const host = value
    .trim()
    .toLowerCase()
    .replace(/^\[|\]$/g, "")
    .replace(/\.+$/, "");
  if (
    host === "" ||
    host === "localhost" ||
    host.endsWith(".localhost") ||
    host === "::" ||
    host === "::1" ||
    host === "0:0:0:0:0:0:0:0" ||
    host === "0:0:0:0:0:0:0:1" ||
    /^::ffff:127\./.test(host)
  ) {
    return false;
  }
  if (host === "0.0.0.0") return false;
  const ipv4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (ipv4 !== null) {
    const octets = ipv4.slice(1).map(Number);
    if (octets.some((part) => part > 255)) return false;
    if (octets[0] === 127 || octets.every((part) => part === 0)) return false;
  }
  return true;
}

function validTimestamp(value: Date): string {
  if (!(value instanceof Date) || !Number.isFinite(value.valueOf())) {
    throw new Error("physical_gpu_campaign_cli_clock_is_invalid");
  }
  return value.toISOString();
}

function redactSecrets(message: string, secrets: readonly string[]): string {
  let result = message;
  for (const secret of [...new Set(secrets)].sort((left, right) => right.length - left.length)) {
    if (secret.length > 0) result = result.replaceAll(secret, "[REDACTED]");
  }
  return result;
}

function normalizeError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

function requireUnique(values: readonly (string | number)[], name: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`physical_gpu_campaign_cli_${name}_is_duplicate`);
  }
}

function sameSet(left: readonly string[], right: readonly string[]): boolean {
  return (
    left.length === right.length &&
    new Set(left).size === left.length &&
    new Set(right).size === right.length &&
    left.every((value) => right.includes(value))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
  name: string,
): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (
    actual.length !== wanted.length ||
    actual.some((key, index) => key !== wanted[index])
  ) {
    throw new Error(`${name}_has_invalid_keys`);
  }
}

function safeInteger(
  value: unknown,
  minimum: number,
  maximum: number,
  name: string,
): number {
  if (
    !Number.isSafeInteger(value) ||
    (value as number) < minimum ||
    (value as number) > maximum
  ) {
    throw new Error(`physical_gpu_campaign_cli_${name}_is_invalid`);
  }
  return value as number;
}

function requiredText(value: unknown, name: string): string {
  if (
    typeof value !== "string" ||
    value.length === 0 ||
    value.length > 4_096 ||
    /[\0\r\n]/.test(value)
  ) {
    throw new Error(`physical_gpu_campaign_cli_${name}_is_invalid`);
  }
  return value;
}

async function main(): Promise<void> {
  process.exitCode = await executePhysicalGpuCampaignCli(process.argv.slice(2));
}

const entryPath = process.argv[1];
if (
  entryPath !== undefined &&
  resolve(fileURLToPath(import.meta.url)) === resolve(entryPath)
) {
  void main().catch((error) => {
    process.stderr.write(`${normalizeError(error).message}\n`);
    process.exitCode = 1;
  });
}
