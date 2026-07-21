import { performance } from "node:perf_hooks";
import {
  PythonLaunchSupervisor,
  type LaunchAgent,
  type LaunchSupervisorOptions,
  type LaunchSupervisorSnapshot,
} from "./launch-supervisor.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
  type PythonRootEngineLaunch,
} from "./python-launcher.js";

export const PHYSICAL_GPU_CAMPAIGN_SCHEMA = "gdlp-physical-gpu-campaign-observation/1" as const;
export const OUTPUT_TOKEN_HASH_SCHEME = "gdlp-output-token-ids-v1" as const;

const AGENT_HEALTH_SCHEMA = "gdlp-launch-agent-health/2";
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const DECIMAL_UINT64_PATTERN = /^(?:0|[1-9][0-9]{0,19})$/;
const MAX_UINT64 = (1n << 64n) - 1n;
const MAX_AGENTS = 64;
const MAX_CANARIES = 256;
const MAX_MESSAGES = 1_024;
const MAX_CONCURRENCIES = 32;
const MAX_CONCURRENCY = 1_024;
const MAX_ITERATIONS = 10_000;
const MAX_WARMUPS = 1_000;
const MAX_TIMEOUT_MS = 86_400_000;

const DEFAULT_TIMEOUTS: Readonly<PhysicalGpuCampaignTimeouts> = Object.freeze({
  agentHealthMs: 10_000,
  apiHealthMs: 10_000,
  apiRequestMs: 300_000,
  sseIdleMs: 30_000,
  cleanupStopMs: 60_000,
});

export type PhysicalGpuCampaignMessageRole = "system" | "developer" | "user" | "assistant";
export type PhysicalGpuCampaignFinishReason = "length" | "stop";
export type PhysicalGpuCampaignSamplePhase = "warmup" | "measure";
export type PhysicalGpuCampaignCanaryPhase = "pre" | "post";
export type PhysicalGpuCampaignAgentHealthPhase = "before" | "after";

export interface PhysicalGpuCampaignMessage {
  role: PhysicalGpuCampaignMessageRole;
  content: string;
}

export interface PhysicalGpuCampaignCanary {
  id: string;
  messages: PhysicalGpuCampaignMessage[];
  maxTokens: number;
  expectedOutputTokenIdsSha256: string;
  expectedCompletionTokens: number;
  expectedFinishReason: PhysicalGpuCampaignFinishReason;
}

export interface PhysicalGpuCampaignAgentBinding {
  nodeId: string;
  agent: LaunchAgent;
}

export interface PhysicalGpuCampaignInput {
  launch: PythonPipelineLaunchDescription;
  agents: PhysicalGpuCampaignAgentBinding[];
  apiBaseUrl: string;
  canaries: PhysicalGpuCampaignCanary[];
  warmups: number;
  iterations: number;
  concurrencies: number[];
}

export interface PhysicalGpuCampaignAgentHealth {
  schema: typeof AGENT_HEALTH_SCHEMA;
  agentId: string;
  nodeId: string | null;
  activeProcesses: number;
  retainedTombstones: number;
}

export interface PhysicalGpuCampaignTimeouts {
  agentHealthMs: number;
  apiHealthMs: number;
  apiRequestMs: number;
  sseIdleMs: number;
  cleanupStopMs: number;
}

export interface PhysicalGpuCampaignSupervisor {
  start(signal?: AbortSignal): Promise<LaunchSupervisorSnapshot>;
  stop(reason?: string): Promise<LaunchSupervisorSnapshot>;
  snapshot(): LaunchSupervisorSnapshot;
}

export type PhysicalGpuCampaignFetch = (
  input: string | URL,
  init?: RequestInit,
) => Promise<Response>;

export type PhysicalGpuCampaignSupervisorFactory = (
  launch: PythonPipelineLaunchDescription,
  options: LaunchSupervisorOptions,
) => PhysicalGpuCampaignSupervisor;

export type PhysicalGpuCampaignHealthReader = (
  binding: PhysicalGpuCampaignAgentBinding,
  signal?: AbortSignal,
) => Promise<unknown>;

export interface PhysicalGpuCampaignDependencies {
  fetch?: PhysicalGpuCampaignFetch;
  now?: () => number;
  supervisor?: PhysicalGpuCampaignSupervisorFactory;
  health?: PhysicalGpuCampaignHealthReader;
  timeouts?: Partial<PhysicalGpuCampaignTimeouts>;
}

export interface PhysicalGpuCampaignLifecycleEvent {
  sequence: number;
  atMs: number;
  phase:
    | "agent_preflight"
    | "supervisor_start"
    | "api_health"
    | "canary_pre"
    | "benchmark"
    | "canary_post"
    | "supervisor_stability"
    | "cleanup_stop"
    | "agent_cleanup";
  passed: boolean;
  detail: string;
}

export interface PhysicalGpuCampaignAgentHealthObservation {
  phase: PhysicalGpuCampaignAgentHealthPhase;
  expectedAgentId: string;
  expectedNodeId: string;
  health: PhysicalGpuCampaignAgentHealth | null;
  passed: boolean;
  error: string | null;
}

export interface PhysicalGpuCampaignApiHealthObservation {
  status: "ready";
  error: null;
  model: string;
  artifactIdentity: string;
  canonicalModelSource: string;
  canonicalModelRevision: string | null;
  pipelineSnapshotIdentity: string;
  stages: number;
  boundaries: number[];
  codec: string;
}

export interface PhysicalGpuCampaignCompletionEvidence {
  promptTokens: number;
  completionTokens: number;
  finishReason: PhysicalGpuCampaignFinishReason;
  outputTokenIdsSha256: string;
  outputTokenIdsHashScheme: typeof OUTPUT_TOKEN_HASH_SCHEME;
  serverTtftMs: number;
  serverTpotMs: number;
  serverPipelineMs: number;
}

export interface PhysicalGpuCampaignCanaryObservation {
  phase: PhysicalGpuCampaignCanaryPhase;
  canaryId: string;
  passed: boolean;
  clientResponseMs: number | null;
  evidence: PhysicalGpuCampaignCompletionEvidence | null;
  error: string | null;
}

export interface PhysicalGpuCampaignRequestSample {
  sampleId: string;
  phase: PhysicalGpuCampaignSamplePhase;
  canaryId: string;
  concurrency: number;
  iteration: number;
  requestIndex: number;
  passed: boolean;
  clientFirstContentMs: number | null;
  clientResponseMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  finishReason: PhysicalGpuCampaignFinishReason | null;
  outputTokenIdsSha256: string | null;
  serverTtftMs: number | null;
  serverTpotMs: number | null;
  serverPipelineMs: number | null;
  perUserOutputTokensPerSecondIncludingTtft: number | null;
  error: string | null;
}

export interface PhysicalGpuCampaignBatchObservation {
  batchId: string;
  phase: PhysicalGpuCampaignSamplePhase;
  concurrency: number;
  iteration: number;
  passed: boolean;
  clientWallMs: number;
  actualCompletionTokens: number;
  aggregateOutputTokensPerSecondIncludingTtft: number | null;
}

export interface PhysicalGpuCampaignMetricStats {
  count: number;
  mean: number | null;
  p50: number | null;
  p95: number | null;
  min: number | null;
  max: number | null;
}

export interface PhysicalGpuCampaignConcurrencySummary {
  concurrency: number;
  measuredRequests: number;
  actualCompletionTokens: number;
  measuredBatchWallMs: number;
  aggregateOutputTokensPerSecondIncludingTtft: number | null;
}

export interface PhysicalGpuCampaignSummary {
  measuredRequests: number;
  actualCompletionTokens: number;
  measuredBatchWallMs: number;
  aggregateOutputTokensPerSecondIncludingTtft: number | null;
  clientFirstContentMs: PhysicalGpuCampaignMetricStats;
  clientResponseMs: PhysicalGpuCampaignMetricStats;
  serverTtftMs: PhysicalGpuCampaignMetricStats;
  serverTpotMs: PhysicalGpuCampaignMetricStats;
  serverPipelineMs: PhysicalGpuCampaignMetricStats;
  byConcurrency: PhysicalGpuCampaignConcurrencySummary[];
}

export interface PhysicalGpuCampaignLifecycleObservation {
  events: PhysicalGpuCampaignLifecycleEvent[];
  agentHealthBefore: PhysicalGpuCampaignAgentHealthObservation[];
  agentHealthAfter: PhysicalGpuCampaignAgentHealthObservation[];
  supervisorStarted: LaunchSupervisorSnapshot | null;
  supervisorBeforeStop: LaunchSupervisorSnapshot | null;
  supervisorStopped: LaunchSupervisorSnapshot | null;
  cleanupAttempted: boolean;
  cleanupPassed: boolean;
}

export interface PhysicalGpuCampaignFailure {
  phase: string;
  message: string;
}

export interface PhysicalGpuCampaignObservation {
  schema: typeof PHYSICAL_GPU_CAMPAIGN_SCHEMA;
  launchId: string;
  pipelineId: string;
  apiBaseUrl: string;
  passed: boolean;
  lifecycle: PhysicalGpuCampaignLifecycleObservation;
  apiHealth: PhysicalGpuCampaignApiHealthObservation | null;
  canaries: {
    pre: PhysicalGpuCampaignCanaryObservation[];
    post: PhysicalGpuCampaignCanaryObservation[];
  };
  samples: PhysicalGpuCampaignRequestSample[];
  batches: PhysicalGpuCampaignBatchObservation[];
  summary: PhysicalGpuCampaignSummary;
  failures: PhysicalGpuCampaignFailure[];
}

interface ValidatedCampaignInput extends PhysicalGpuCampaignInput {
  expectedHealth: {
    model: string;
    artifactIdentity: string;
    canonicalModelSource: string;
    canonicalModelRevision: string | null;
    pipelineSnapshotIdentity: string;
    stages: number;
    boundaries: number[];
    codec: string;
  };
}

interface SseResult {
  document: Record<string, unknown>;
  firstContentAtMs: number;
  doneAtMs: number;
}

export async function runPhysicalGpuCampaign(
  inputValue: unknown,
  dependencies: PhysicalGpuCampaignDependencies = {},
): Promise<PhysicalGpuCampaignObservation> {
  const input = validateCampaignInput(inputValue);
  const fetchImpl = dependencies.fetch ?? globalThis.fetch.bind(globalThis);
  const now = checkedClock(dependencies.now ?? performance.now.bind(performance));
  const supervisorFactory =
    dependencies.supervisor ??
    ((launch, options) => new PythonLaunchSupervisor(launch, options));
  const healthReader = dependencies.health ?? defaultAgentHealthReader;
  const timeouts = normalizeTimeouts(dependencies.timeouts);
  assertDependencies(fetchImpl, supervisorFactory, healthReader);

  const lifecycle: PhysicalGpuCampaignLifecycleObservation = {
    events: [],
    agentHealthBefore: [],
    agentHealthAfter: [],
    supervisorStarted: null,
    supervisorBeforeStop: null,
    supervisorStopped: null,
    cleanupAttempted: false,
    cleanupPassed: false,
  };
  const observation: PhysicalGpuCampaignObservation = {
    schema: PHYSICAL_GPU_CAMPAIGN_SCHEMA,
    launchId: input.launch.launchId,
    pipelineId: input.launch.pipelineId,
    apiBaseUrl: input.apiBaseUrl,
    passed: false,
    lifecycle,
    apiHealth: null,
    canaries: { pre: [], post: [] },
    samples: [],
    batches: [],
    summary: summarize([], []),
    failures: [],
  };
  let eventSequence = 0;
  let supervisor: PhysicalGpuCampaignSupervisor | null = null;
  let corePassed = false;
  const agentByNode = new Map(input.agents.map((binding) => [binding.nodeId, binding.agent]));
  const recordEvent = (
    phase: PhysicalGpuCampaignLifecycleEvent["phase"],
    passed: boolean,
    detail: string,
  ): void => {
    lifecycle.events.push({ sequence: eventSequence++, atMs: now(), phase, passed, detail });
  };
  const recordFailure = (phase: string, error: unknown): void => {
    observation.failures.push({ phase, message: normalizeError(error).message });
  };

  try {
    lifecycle.agentHealthBefore = await collectAgentHealth(
      input.agents,
      "before",
      healthReader,
      timeouts.agentHealthMs,
    );
    const preflightPassed = lifecycle.agentHealthBefore.every((item) => item.passed);
    recordEvent(
      "agent_preflight",
      preflightPassed,
      preflightPassed ? "all_launch_agents_inactive" : "launch_agent_preflight_failed",
    );
    if (!preflightPassed) throw new Error("physical_gpu_campaign_agent_preflight_failed");

    supervisor = supervisorFactory(input.launch, {
      resolveAgent: (nodeId) => agentByNode.get(nodeId),
      now,
    });
    const started = await supervisor.start();
    assertRunningSupervisor(started, input.launch);
    lifecycle.supervisorStarted = structuredClone(started);
    recordEvent("supervisor_start", true, "all_launch_processes_ready");

    observation.apiHealth = await observeApiHealth(
      fetchImpl,
      input.apiBaseUrl,
      input.expectedHealth,
      timeouts.apiHealthMs,
    );
    recordEvent("api_health", true, "root_api_identity_exact");

    observation.canaries.pre = await runCanaryCorpus(
      "pre",
      input,
      fetchImpl,
      now,
      timeouts.apiRequestMs,
    );
    const preCanariesPassed = observation.canaries.pre.every((item) => item.passed);
    recordEvent(
      "canary_pre",
      preCanariesPassed,
      preCanariesPassed ? "pre_canaries_exact" : "pre_canary_failed",
    );
    if (!preCanariesPassed) {
      for (const item of observation.canaries.pre.filter((item) => !item.passed)) {
        recordFailure("canary_pre", item.error ?? `canary_failed:${item.canaryId}`);
      }
      throw new Error("physical_gpu_campaign_pre_canary_failed");
    }

    const benchmarkPassed = await runBenchmark(
      input,
      fetchImpl,
      now,
      observation,
      timeouts.apiRequestMs,
      timeouts.sseIdleMs,
    );
    recordEvent(
      "benchmark",
      benchmarkPassed,
      benchmarkPassed ? "all_sse_samples_valid" : "sse_benchmark_failed",
    );
    if (!benchmarkPassed) {
      for (const sample of observation.samples.filter((item) => !item.passed)) {
        recordFailure("benchmark", sample.error ?? `sample_failed:${sample.sampleId}`);
      }
    }

    observation.canaries.post = await runCanaryCorpus(
      "post",
      input,
      fetchImpl,
      now,
      timeouts.apiRequestMs,
    );
    const postCanariesPassed = observation.canaries.post.every((item) => item.passed);
    recordEvent(
      "canary_post",
      postCanariesPassed,
      postCanariesPassed ? "post_canaries_exact" : "post_canary_failed",
    );
    if (!postCanariesPassed) {
      for (const item of observation.canaries.post.filter((item) => !item.passed)) {
        recordFailure("canary_post", item.error ?? `canary_failed:${item.canaryId}`);
      }
    }

    const beforeStop = supervisor.snapshot();
    assertRunningSupervisor(beforeStop, input.launch);
    lifecycle.supervisorBeforeStop = structuredClone(beforeStop);
    recordEvent("supervisor_stability", true, "all_processes_still_ready");
    corePassed = benchmarkPassed && postCanariesPassed;
  } catch (error) {
    recordFailure("campaign", error);
    const last = lifecycle.events.at(-1);
    if (last?.passed !== false) {
      const phase = supervisor === null ? "supervisor_start" : "supervisor_stability";
      recordEvent(phase, false, normalizeError(error).message);
    }
  } finally {
    let stopPassed = supervisor === null;
    if (supervisor !== null) {
      lifecycle.cleanupAttempted = true;
      try {
        lifecycle.supervisorBeforeStop ??= structuredClone(supervisor.snapshot());
        const stopped = await withCampaignDeadline(
          "cleanup_stop",
          timeouts.cleanupStopMs,
          () => supervisor!.stop("physical_gpu_campaign_complete"),
        );
        lifecycle.supervisorStopped = structuredClone(stopped);
        stopPassed = isCleanStoppedSupervisor(stopped, input.launch);
        recordEvent(
          "cleanup_stop",
          stopPassed,
          stopPassed ? "all_supervised_processes_stopped" : "supervisor_cleanup_incomplete",
        );
        if (!stopPassed) recordFailure("cleanup_stop", "supervisor_cleanup_incomplete");
      } catch (error) {
        recordFailure("cleanup_stop", error);
        recordEvent("cleanup_stop", false, normalizeError(error).message);
      }
    }

    lifecycle.agentHealthAfter = await collectAgentHealth(
      input.agents,
      "after",
      healthReader,
      timeouts.agentHealthMs,
    );
    const agentsClean = lifecycle.agentHealthAfter.every((item) => item.passed);
    recordEvent(
      "agent_cleanup",
      agentsClean,
      agentsClean ? "all_launch_agents_inactive" : "residual_agent_processes_detected",
    );
    if (!agentsClean) recordFailure("agent_cleanup", "residual_agent_processes_detected");
    lifecycle.cleanupPassed = stopPassed && agentsClean;
  }

  observation.summary = summarize(observation.samples, observation.batches);
  observation.passed = corePassed && lifecycle.cleanupPassed && observation.failures.length === 0;
  return observation;
}

function validateCampaignInput(value: unknown): ValidatedCampaignInput {
  const input = object(value, "physical_gpu_campaign_input");
  exactKeys(
    input,
    ["launch", "agents", "apiBaseUrl", "canaries", "warmups", "iterations", "concurrencies"],
    "physical_gpu_campaign_input",
  );
  try {
    validatePythonLaunchDescription(input.launch);
  } catch (error) {
    throw new Error("physical_gpu_campaign_launch_is_invalid", { cause: error });
  }
  const launch = structuredClone(input.launch);
  const root = rootLaunch(launch);
  const launchNodeIds = unique(launch.launchOrder.map((process) => process.anchor.memberId));
  if (launchNodeIds.length < 2) throw new Error("physical_gpu_campaign_requires_two_launch_nodes");

  const agentsValue = array(input.agents, 2, MAX_AGENTS, "physical_gpu_campaign_agents");
  const agents = agentsValue.map((item, index) => validateAgentBinding(item, index));
  assertUnique(agents.map((item) => item.nodeId), "physical_gpu_campaign_agent_node_is_duplicate");
  assertUnique(agents.map((item) => item.agent.id), "physical_gpu_campaign_agent_id_is_duplicate");
  if (!sameSet(launchNodeIds, agents.map((item) => item.nodeId))) {
    throw new Error("physical_gpu_campaign_agents_do_not_match_launch_nodes");
  }

  const apiBaseUrl = normalizeBaseUrl(input.apiBaseUrl);
  const warmups = integer(input.warmups, 1, MAX_WARMUPS, "physical_gpu_campaign_warmups");
  const iterations = integer(
    input.iterations,
    5,
    MAX_ITERATIONS,
    "physical_gpu_campaign_iterations",
  );
  const concurrencies = array(
    input.concurrencies,
    1,
    MAX_CONCURRENCIES,
    "physical_gpu_campaign_concurrencies",
  ).map((item) =>
    integer(item, 1, MAX_CONCURRENCY, "physical_gpu_campaign_concurrency"),
  );
  assertUnique(concurrencies, "physical_gpu_campaign_concurrency_is_duplicate");

  const canaries = array(input.canaries, 3, MAX_CANARIES, "physical_gpu_campaign_canaries").map(
    (item, index) => validateCanary(item, index, launch.configuration.maxOutputTokens),
  );
  assertUnique(canaries.map((item) => item.id), "physical_gpu_campaign_canary_id_is_duplicate");

  const artifactIdentity = requiredText(
    launch.runtimeModel.artifactIdentity,
    "physical_gpu_campaign_artifact_identity",
  );
  const canonicalModelSource = requiredText(
    launch.runtimeModel.canonicalSource,
    "physical_gpu_campaign_canonical_model_source",
  );
  const pipelineSnapshotIdentity = decimalUint64(
    launch.runtimeModel.snapshotIdentity,
    "physical_gpu_campaign_pipeline_snapshot_identity",
  );
  const canonicalModelRevision = launch.runtimeModel.canonicalRevision ?? null;
  if (canonicalModelRevision !== null) {
    requiredText(canonicalModelRevision, "physical_gpu_campaign_canonical_model_revision");
  }
  const expectedHealth = {
    model: launch.configuration.publicModelName,
    artifactIdentity,
    canonicalModelSource,
    canonicalModelRevision,
    pipelineSnapshotIdentity,
    stages: root.boundaries.length - 1,
    boundaries: [...root.boundaries],
    codec: launch.route.codec,
  };
  return { launch, agents, apiBaseUrl, canaries, warmups, iterations, concurrencies, expectedHealth };
}

function validateAgentBinding(value: unknown, index: number): PhysicalGpuCampaignAgentBinding {
  const binding = object(value, `physical_gpu_campaign_agent_${index}`);
  exactKeys(binding, ["nodeId", "agent"], `physical_gpu_campaign_agent_${index}`);
  const nodeId = identifier(binding.nodeId, `physical_gpu_campaign_agent_${index}_node`);
  const agent = binding.agent;
  if (typeof agent !== "object" || agent === null) {
    throw new Error(`physical_gpu_campaign_agent_${index}_is_invalid`);
  }
  const candidate = agent as Partial<LaunchAgent>;
  identifier(candidate.id, `physical_gpu_campaign_agent_${index}_id`);
  if (typeof candidate.start !== "function") {
    throw new Error(`physical_gpu_campaign_agent_${index}_start_is_invalid`);
  }
  return { nodeId, agent: candidate as LaunchAgent };
}

function validateCanary(
  value: unknown,
  index: number,
  maximumOutputTokens: number,
): PhysicalGpuCampaignCanary {
  const canary = object(value, `physical_gpu_campaign_canary_${index}`);
  exactKeys(
    canary,
    [
      "id",
      "messages",
      "maxTokens",
      "expectedOutputTokenIdsSha256",
      "expectedCompletionTokens",
      "expectedFinishReason",
    ],
    `physical_gpu_campaign_canary_${index}`,
  );
  const id = identifier(canary.id, `physical_gpu_campaign_canary_${index}_id`);
  const messages = array(
    canary.messages,
    1,
    MAX_MESSAGES,
    `physical_gpu_campaign_canary_${index}_messages`,
  ).map((item, messageIndex) => {
    const message = object(item, `physical_gpu_campaign_canary_${index}_message_${messageIndex}`);
    exactKeys(
      message,
      ["role", "content"],
      `physical_gpu_campaign_canary_${index}_message_${messageIndex}`,
    );
    const role = oneOf(
      message.role,
      ["system", "developer", "user", "assistant"] as const,
      `physical_gpu_campaign_canary_${index}_message_${messageIndex}_role`,
    );
    const content = requiredText(
      message.content,
      `physical_gpu_campaign_canary_${index}_message_${messageIndex}_content`,
      1_000_000,
    );
    return { role, content };
  });
  const maxTokens = integer(
    canary.maxTokens,
    1,
    maximumOutputTokens,
    `physical_gpu_campaign_canary_${index}_max_tokens`,
  );
  const expectedCompletionTokens = integer(
    canary.expectedCompletionTokens,
    1,
    maxTokens,
    `physical_gpu_campaign_canary_${index}_completion_tokens`,
  );
  const expectedFinishReason = oneOf(
    canary.expectedFinishReason,
    ["length", "stop"] as const,
    `physical_gpu_campaign_canary_${index}_finish_reason`,
  );
  if (expectedFinishReason === "length" && expectedCompletionTokens !== maxTokens) {
    throw new Error(`physical_gpu_campaign_canary_${index}_length_count_mismatch`);
  }
  const expectedOutputTokenIdsSha256 = sha256(
    canary.expectedOutputTokenIdsSha256,
    `physical_gpu_campaign_canary_${index}_token_hash`,
  );
  return {
    id,
    messages,
    maxTokens,
    expectedOutputTokenIdsSha256,
    expectedCompletionTokens,
    expectedFinishReason,
  };
}

async function collectAgentHealth(
  bindings: PhysicalGpuCampaignAgentBinding[],
  phase: PhysicalGpuCampaignAgentHealthPhase,
  reader: PhysicalGpuCampaignHealthReader,
  timeoutMs: number,
): Promise<PhysicalGpuCampaignAgentHealthObservation[]> {
  return Promise.all(bindings.map(async (binding) => {
    try {
      const health = validateAgentHealth(
        await withCampaignDeadline(
          `agent_health:${phase}:${binding.nodeId}`,
          timeoutMs,
          (signal) => reader(binding, signal),
        ),
      );
      const passed =
        health.agentId === binding.agent.id &&
        health.nodeId === binding.nodeId &&
        health.activeProcesses === 0;
      return {
        phase,
        expectedAgentId: binding.agent.id,
        expectedNodeId: binding.nodeId,
        health,
        passed,
        error: passed ? null : "launch_agent_health_identity_or_process_mismatch",
      };
    } catch (error) {
      return {
        phase,
        expectedAgentId: binding.agent.id,
        expectedNodeId: binding.nodeId,
        health: null,
        passed: false,
        error: normalizeError(error).message,
      };
    }
  }));
}

async function defaultAgentHealthReader(
  binding: PhysicalGpuCampaignAgentBinding,
  signal?: AbortSignal,
): Promise<unknown> {
  const candidate = binding.agent as LaunchAgent & {
    health?: (signal?: AbortSignal) => Promise<unknown>;
  };
  if (typeof candidate.health !== "function") {
    throw new Error(`physical_gpu_campaign_agent_health_not_available:${binding.nodeId}`);
  }
  return candidate.health.call(binding.agent, signal);
}

function validateAgentHealth(value: unknown): PhysicalGpuCampaignAgentHealth {
  const health = object(value, "physical_gpu_campaign_agent_health");
  exactKeys(
    health,
    ["schema", "agentId", "nodeId", "activeProcesses", "retainedTombstones"],
    "physical_gpu_campaign_agent_health",
  );
  if (health.schema !== AGENT_HEALTH_SCHEMA) {
    throw new Error("physical_gpu_campaign_agent_health_schema_mismatch");
  }
  const agentId = identifier(health.agentId, "physical_gpu_campaign_agent_health_agent_id");
  const nodeId =
    health.nodeId === null
      ? null
      : identifier(health.nodeId, "physical_gpu_campaign_agent_health_node_id");
  const activeProcesses = integer(
    health.activeProcesses,
    0,
    Number.MAX_SAFE_INTEGER,
    "physical_gpu_campaign_agent_health_active_processes",
  );
  const retainedTombstones = integer(
    health.retainedTombstones,
    0,
    Number.MAX_SAFE_INTEGER,
    "physical_gpu_campaign_agent_health_retained_tombstones",
  );
  return {
    schema: AGENT_HEALTH_SCHEMA,
    agentId,
    nodeId,
    activeProcesses,
    retainedTombstones,
  };
}

async function observeApiHealth(
  fetchImpl: PhysicalGpuCampaignFetch,
  apiBaseUrl: string,
  expected: ValidatedCampaignInput["expectedHealth"],
  timeoutMs: number,
): Promise<PhysicalGpuCampaignApiHealthObservation> {
  return withCampaignDeadline("api_health", timeoutMs, async (signal) => {
    const response = await fetchImpl(`${apiBaseUrl}/health`, {
      method: "GET",
      headers: { accept: "application/json" },
      signal,
    });
    if (response.status !== 200) {
      throw new Error(`physical_gpu_campaign_api_health_http_status:${response.status}`);
    }
    const health = await responseJsonObject(response, "physical_gpu_campaign_api_health");
    if (health.status !== "ready" || health.error !== null) {
      throw new Error("physical_gpu_campaign_api_is_not_ready");
    }
    const observed: PhysicalGpuCampaignApiHealthObservation = {
      status: "ready",
      error: null,
      model: requiredText(health.model, "physical_gpu_campaign_api_health_model"),
      artifactIdentity: requiredText(
        health.artifact_identity,
        "physical_gpu_campaign_api_health_artifact_identity",
      ),
      canonicalModelSource: requiredText(
        health.canonical_model_source,
        "physical_gpu_campaign_api_health_canonical_source",
      ),
      canonicalModelRevision:
        health.canonical_model_revision === null
          ? null
          : requiredText(
              health.canonical_model_revision,
              "physical_gpu_campaign_api_health_canonical_revision",
            ),
      pipelineSnapshotIdentity: decimalUint64(
        health.pipeline_snapshot_identity,
        "physical_gpu_campaign_api_health_pipeline_identity",
      ),
      stages: integer(
        health.stages,
        1,
        1_024,
        "physical_gpu_campaign_api_health_stages",
      ),
      boundaries: integerArray(
        health.boundaries,
        2,
        1_025,
        "physical_gpu_campaign_api_health_boundaries",
      ),
      codec: requiredText(health.codec, "physical_gpu_campaign_api_health_codec"),
    };
    if (
      observed.model !== expected.model ||
      observed.artifactIdentity !== expected.artifactIdentity ||
      observed.canonicalModelSource !== expected.canonicalModelSource ||
      observed.canonicalModelRevision !== expected.canonicalModelRevision ||
      observed.pipelineSnapshotIdentity !== expected.pipelineSnapshotIdentity ||
      observed.stages !== expected.stages ||
      !sameNumberArray(observed.boundaries, expected.boundaries) ||
      observed.codec !== expected.codec
    ) {
      throw new Error("physical_gpu_campaign_api_health_identity_mismatch");
    }
    return observed;
  });
}

async function runCanaryCorpus(
  phase: PhysicalGpuCampaignCanaryPhase,
  input: ValidatedCampaignInput,
  fetchImpl: PhysicalGpuCampaignFetch,
  now: () => number,
  timeoutMs: number,
): Promise<PhysicalGpuCampaignCanaryObservation[]> {
  const observations: PhysicalGpuCampaignCanaryObservation[] = [];
  for (const canary of input.canaries) {
    const startedAt = now();
    try {
      const { evidence, clientResponseMs } = await withCampaignDeadline(
        `canary:${phase}:${canary.id}`,
        timeoutMs,
        async (signal) => {
          const response = await fetchImpl(`${input.apiBaseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { accept: "application/json", "content-type": "application/json" },
            body: JSON.stringify(requestBody(input.expectedHealth.model, canary, false)),
            signal,
          });
          if (response.status !== 200) {
            throw new Error(`physical_gpu_campaign_canary_http_status:${response.status}`);
          }
          const document = await responseJsonObject(
            response,
            "physical_gpu_campaign_canary_response",
          );
          return {
            evidence: completionEvidence(document, input.expectedHealth.model, canary),
            clientResponseMs: duration(
              startedAt,
              now(),
              "physical_gpu_campaign_canary_response",
            ),
          };
        },
      );
      observations.push({
        phase,
        canaryId: canary.id,
        passed: true,
        clientResponseMs,
        evidence,
        error: null,
      });
    } catch (error) {
      observations.push({
        phase,
        canaryId: canary.id,
        passed: false,
        clientResponseMs: null,
        evidence: null,
        error: normalizeError(error).message,
      });
    }
  }
  return observations;
}

async function runBenchmark(
  input: ValidatedCampaignInput,
  fetchImpl: PhysicalGpuCampaignFetch,
  now: () => number,
  observation: PhysicalGpuCampaignObservation,
  requestTimeoutMs: number,
  sseIdleTimeoutMs: number,
): Promise<boolean> {
  for (const concurrency of input.concurrencies) {
    for (const phase of ["warmup", "measure"] as const) {
      const count = phase === "warmup" ? input.warmups : input.iterations;
      for (let iteration = 0; iteration < count; iteration += 1) {
        const batchId = `${phase}-c${concurrency}-i${iteration}`;
        const batchStartedAt = now();
        const samples = await Promise.all(
          Array.from({ length: concurrency }, (_, requestIndex) => {
            const canary =
              input.canaries[(iteration * concurrency + requestIndex) % input.canaries.length]!;
            return runSseSample(
              input,
              canary,
              phase,
              concurrency,
              iteration,
              requestIndex,
              batchId,
              fetchImpl,
              now,
              requestTimeoutMs,
              sseIdleTimeoutMs,
            );
          }),
        );
        const clientWallMs = duration(
          batchStartedAt,
          now(),
          "physical_gpu_campaign_batch_wall",
        );
        const actualCompletionTokens = samples.reduce(
          (sum, sample) => sum + (sample.completionTokens ?? 0),
          0,
        );
        const passed = samples.every((sample) => sample.passed);
        observation.samples.push(...samples);
        observation.batches.push({
          batchId,
          phase,
          concurrency,
          iteration,
          passed,
          clientWallMs,
          actualCompletionTokens,
          aggregateOutputTokensPerSecondIncludingTtft:
            clientWallMs > 0 ? actualCompletionTokens / (clientWallMs / 1_000) : null,
        });
        if (!passed || clientWallMs <= 0) return false;
      }
    }
  }
  return true;
}

async function runSseSample(
  input: ValidatedCampaignInput,
  canary: PhysicalGpuCampaignCanary,
  phase: PhysicalGpuCampaignSamplePhase,
  concurrency: number,
  iteration: number,
  requestIndex: number,
  batchId: string,
  fetchImpl: PhysicalGpuCampaignFetch,
  now: () => number,
  requestTimeoutMs: number,
  sseIdleTimeoutMs: number,
): Promise<PhysicalGpuCampaignRequestSample> {
  const sampleId = `${batchId}-r${requestIndex}`;
  const startedAt = now();
  try {
    const { evidence, clientFirstContentMs, clientResponseMs } =
      await withCampaignDeadline(
        `sse_request:${sampleId}`,
        requestTimeoutMs,
        async (signal) => {
          const response = await fetchImpl(`${input.apiBaseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { accept: "text/event-stream", "content-type": "application/json" },
            body: JSON.stringify(requestBody(input.expectedHealth.model, canary, true)),
            signal,
          });
          if (response.status !== 200) {
            throw new Error(`physical_gpu_campaign_sse_http_status:${response.status}`);
          }
          const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
          if (!contentType.includes("text/event-stream")) {
            throw new Error("physical_gpu_campaign_sse_content_type_mismatch");
          }
          const streamed = await readSse(response, now, signal, sseIdleTimeoutMs);
          return {
            evidence: completionEvidence(streamed.document, input.expectedHealth.model, canary),
            clientFirstContentMs: duration(
              startedAt,
              streamed.firstContentAtMs,
              "physical_gpu_campaign_first_content",
            ),
            clientResponseMs: duration(
              startedAt,
              streamed.doneAtMs,
              "physical_gpu_campaign_response",
            ),
          };
        },
      );
    if (clientResponseMs <= 0) throw new Error("physical_gpu_campaign_response_duration_is_zero");
    return {
      sampleId,
      phase,
      canaryId: canary.id,
      concurrency,
      iteration,
      requestIndex,
      passed: true,
      clientFirstContentMs,
      clientResponseMs,
      promptTokens: evidence.promptTokens,
      completionTokens: evidence.completionTokens,
      finishReason: evidence.finishReason,
      outputTokenIdsSha256: evidence.outputTokenIdsSha256,
      serverTtftMs: evidence.serverTtftMs,
      serverTpotMs: evidence.serverTpotMs,
      serverPipelineMs: evidence.serverPipelineMs,
      perUserOutputTokensPerSecondIncludingTtft:
        evidence.completionTokens / (clientResponseMs / 1_000),
      error: null,
    };
  } catch (error) {
    return {
      sampleId,
      phase,
      canaryId: canary.id,
      concurrency,
      iteration,
      requestIndex,
      passed: false,
      clientFirstContentMs: null,
      clientResponseMs: null,
      promptTokens: null,
      completionTokens: null,
      finishReason: null,
      outputTokenIdsSha256: null,
      serverTtftMs: null,
      serverTpotMs: null,
      serverPipelineMs: null,
      perUserOutputTokensPerSecondIncludingTtft: null,
      error: normalizeError(error).message,
    };
  }
}

async function readSse(
  response: Response,
  now: () => number,
  signal: AbortSignal,
  idleTimeoutMs: number,
): Promise<SseResult> {
  if (response.body === null) throw new Error("physical_gpu_campaign_sse_body_missing");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  let firstContentAtMs: number | null = null;
  let doneAtMs: number | null = null;
  let finalDocument: Record<string, unknown> | null = null;
  let doneSeen = false;
  try {
    while (true) {
      const { done, value } = await readSseChunk(reader, signal, idleTimeoutMs);
      buffer += decoder.decode(value, { stream: !done });
      buffer = buffer.replaceAll("\r\n", "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const event = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = event
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).replace(/^ /, ""))
          .join("\n");
        if (data !== "") {
          if (doneSeen) throw new Error("physical_gpu_campaign_sse_event_after_done");
          if (data === "[DONE]") {
            doneSeen = true;
            doneAtMs = now();
          } else {
            const document = jsonObject(data, "physical_gpu_campaign_sse_event");
            if ("error" in document) throw new Error("physical_gpu_campaign_sse_error_event");
            const choice = firstChoice(document, "physical_gpu_campaign_sse_choice");
            const delta = object(choice.delta, "physical_gpu_campaign_sse_delta");
            if (
              typeof delta.content === "string" &&
              delta.content.length > 0 &&
              firstContentAtMs === null
            ) {
              firstContentAtMs = now();
            }
            if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
              if (finalDocument !== null) {
                throw new Error("physical_gpu_campaign_sse_multiple_final_chunks");
              }
              finalDocument = document;
            }
          }
        }
        boundary = buffer.indexOf("\n\n");
      }
      if (done) break;
    }
    if (buffer.trim() !== "") throw new Error("physical_gpu_campaign_sse_trailing_partial_event");
    if (!doneSeen || doneAtMs === null) throw new Error("physical_gpu_campaign_sse_done_missing");
    if (finalDocument === null) throw new Error("physical_gpu_campaign_sse_final_chunk_missing");
    if (firstContentAtMs === null) throw new Error("physical_gpu_campaign_sse_first_content_missing");
    return { document: finalDocument, firstContentAtMs, doneAtMs };
  } catch (error) {
    try {
      await reader.cancel(error);
    } catch {
      // Cancellation is best-effort; preserve the deadline/protocol error.
    }
    throw error;
  } finally {
    reader.releaseLock();
  }
}

async function readSseChunk(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  signal: AbortSignal,
  idleTimeoutMs: number,
): Promise<{ done: boolean; value: Uint8Array | undefined }> {
  if (signal.aborted) throw campaignCancellation(signal);
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (callback: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      callback();
    };
    const timeoutError = new Error(`physical_gpu_campaign_timeout:sse_idle:${idleTimeoutMs}`);
    const abort = () => finish(() => reject(campaignCancellation(signal)));
    const timer = setTimeout(() => finish(() => reject(timeoutError)), idleTimeoutMs);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) {
      abort();
      return;
    }
    reader.read().then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(normalizeError(error))),
    );
  });
}

function completionEvidence(
  document: Record<string, unknown>,
  expectedModel: string,
  canary: PhysicalGpuCampaignCanary,
): PhysicalGpuCampaignCompletionEvidence {
  if (document.model !== expectedModel) throw new Error("physical_gpu_campaign_completion_model_mismatch");
  const choice = firstChoice(document, "physical_gpu_campaign_completion_choice");
  const finishReason = oneOf(
    choice.finish_reason,
    ["length", "stop"] as const,
    "physical_gpu_campaign_completion_finish_reason",
  );
  const usage = object(document.usage, "physical_gpu_campaign_completion_usage");
  const promptTokens = integer(
    usage.prompt_tokens,
    1,
    Number.MAX_SAFE_INTEGER,
    "physical_gpu_campaign_completion_prompt_tokens",
  );
  const completionTokens = integer(
    usage.completion_tokens,
    1,
    Number.MAX_SAFE_INTEGER,
    "physical_gpu_campaign_completion_tokens",
  );
  const totalTokens = integer(
    usage.total_tokens,
    1,
    Number.MAX_SAFE_INTEGER,
    "physical_gpu_campaign_completion_total_tokens",
  );
  if (totalTokens !== promptTokens + completionTokens) {
    throw new Error("physical_gpu_campaign_completion_usage_mismatch");
  }
  const metrics = object(
    document.distribution_metrics,
    "physical_gpu_campaign_distribution_metrics",
  );
  const outputTokenIdsSha256 = sha256(
    metrics.output_token_ids_sha256,
    "physical_gpu_campaign_output_token_hash",
  );
  if (metrics.output_token_ids_hash_scheme !== OUTPUT_TOKEN_HASH_SCHEME) {
    throw new Error("physical_gpu_campaign_output_token_hash_scheme_mismatch");
  }
  const serverTtftMs = nonNegative(metrics.ttft_ms, "physical_gpu_campaign_server_ttft");
  const serverTpotMs = nonNegative(metrics.tpot_ms, "physical_gpu_campaign_server_tpot");
  const serverPipelineMs = nonNegative(
    metrics.pipeline_ms,
    "physical_gpu_campaign_server_pipeline",
  );
  const reconstructedPipelineMs = serverTtftMs + Math.max(0, completionTokens - 1) * serverTpotMs;
  const tolerance = Math.max(0.5, serverPipelineMs * 0.005);
  if (
    serverPipelineMs + tolerance < serverTtftMs ||
    Math.abs(serverPipelineMs - reconstructedPipelineMs) > tolerance
  ) {
    throw new Error("physical_gpu_campaign_server_metrics_are_inconsistent");
  }
  if (
    completionTokens !== canary.expectedCompletionTokens ||
    finishReason !== canary.expectedFinishReason ||
    outputTokenIdsSha256 !== canary.expectedOutputTokenIdsSha256
  ) {
    throw new Error(`physical_gpu_campaign_exact_parity_failed:${canary.id}`);
  }
  return {
    promptTokens,
    completionTokens,
    finishReason,
    outputTokenIdsSha256,
    outputTokenIdsHashScheme: OUTPUT_TOKEN_HASH_SCHEME,
    serverTtftMs,
    serverTpotMs,
    serverPipelineMs,
  };
}

function summarize(
  samples: PhysicalGpuCampaignRequestSample[],
  batches: PhysicalGpuCampaignBatchObservation[],
): PhysicalGpuCampaignSummary {
  const measuredSamples = samples.filter((sample) => sample.phase === "measure" && sample.passed);
  const measuredBatches = batches.filter((batch) => batch.phase === "measure" && batch.passed);
  const actualCompletionTokens = measuredSamples.reduce(
    (sum, sample) => sum + (sample.completionTokens ?? 0),
    0,
  );
  const measuredBatchWallMs = measuredBatches.reduce((sum, batch) => sum + batch.clientWallMs, 0);
  const byConcurrency = unique(measuredBatches.map((batch) => batch.concurrency)).map(
    (concurrency) => {
      const concurrencySamples = measuredSamples.filter((sample) => sample.concurrency === concurrency);
      const concurrencyBatches = measuredBatches.filter((batch) => batch.concurrency === concurrency);
      const tokens = concurrencySamples.reduce(
        (sum, sample) => sum + (sample.completionTokens ?? 0),
        0,
      );
      const wall = concurrencyBatches.reduce((sum, batch) => sum + batch.clientWallMs, 0);
      return {
        concurrency,
        measuredRequests: concurrencySamples.length,
        actualCompletionTokens: tokens,
        measuredBatchWallMs: wall,
        aggregateOutputTokensPerSecondIncludingTtft: wall > 0 ? tokens / (wall / 1_000) : null,
      };
    },
  );
  return {
    measuredRequests: measuredSamples.length,
    actualCompletionTokens,
    measuredBatchWallMs,
    aggregateOutputTokensPerSecondIncludingTtft:
      measuredBatchWallMs > 0 ? actualCompletionTokens / (measuredBatchWallMs / 1_000) : null,
    clientFirstContentMs: stats(
      measuredSamples.map((sample) => sample.clientFirstContentMs).filter(isNumber),
    ),
    clientResponseMs: stats(
      measuredSamples.map((sample) => sample.clientResponseMs).filter(isNumber),
    ),
    serverTtftMs: stats(measuredSamples.map((sample) => sample.serverTtftMs).filter(isNumber)),
    serverTpotMs: stats(measuredSamples.map((sample) => sample.serverTpotMs).filter(isNumber)),
    serverPipelineMs: stats(
      measuredSamples.map((sample) => sample.serverPipelineMs).filter(isNumber),
    ),
    byConcurrency,
  };
}

function stats(values: number[]): PhysicalGpuCampaignMetricStats {
  if (values.length === 0) {
    return { count: 0, mean: null, p50: null, p95: null, min: null, max: null };
  }
  const ordered = [...values].sort((left, right) => left - right);
  return {
    count: ordered.length,
    mean: ordered.reduce((sum, value) => sum + value, 0) / ordered.length,
    p50: percentile(ordered, 0.5),
    p95: percentile(ordered, 0.95),
    min: ordered[0]!,
    max: ordered.at(-1)!,
  };
}

function percentile(ordered: number[], fraction: number): number {
  const position = (ordered.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower]!;
  const weight = position - lower;
  return ordered[lower]! * (1 - weight) + ordered[upper]! * weight;
}

function requestBody(model: string, canary: PhysicalGpuCampaignCanary, stream: boolean) {
  return {
    model,
    messages: canary.messages,
    max_tokens: canary.maxTokens,
    temperature: 0,
    top_p: 1,
    stream,
  };
}

function assertRunningSupervisor(
  snapshot: LaunchSupervisorSnapshot,
  launch: PythonPipelineLaunchDescription,
): void {
  if (snapshot.state !== "running" || snapshot.failure !== null) {
    throw new Error("physical_gpu_campaign_supervisor_is_not_running");
  }
  const expected = launch.launchOrder.map((process) => process.processId);
  const observed = snapshot.processes.map((process) => process.processId);
  if (!sameSet(expected, observed) || snapshot.processes.some((process) => process.state !== "ready")) {
    throw new Error("physical_gpu_campaign_supervisor_process_readiness_mismatch");
  }
}

function isCleanStoppedSupervisor(
  snapshot: LaunchSupervisorSnapshot,
  launch: PythonPipelineLaunchDescription,
): boolean {
  return (
    snapshot.state === "stopped" &&
    snapshot.failure === null &&
    sameSet(
      launch.launchOrder.map((process) => process.processId),
      snapshot.processes.map((process) => process.processId),
    ) &&
    snapshot.processes.every((process) => process.state === "stopped")
  );
}

function rootLaunch(launch: PythonPipelineLaunchDescription): PythonRootEngineLaunch {
  const roots = launch.launchOrder.filter(
    (process): process is PythonRootEngineLaunch => process.kind === "root-engine",
  );
  if (roots.length !== 1) throw new Error("physical_gpu_campaign_root_launch_is_invalid");
  return roots[0]!;
}

function assertDependencies(
  fetchImpl: PhysicalGpuCampaignFetch,
  supervisor: PhysicalGpuCampaignSupervisorFactory,
  health: PhysicalGpuCampaignHealthReader,
): void {
  if (typeof fetchImpl !== "function") throw new Error("physical_gpu_campaign_fetch_is_invalid");
  if (typeof supervisor !== "function") {
    throw new Error("physical_gpu_campaign_supervisor_factory_is_invalid");
  }
  if (typeof health !== "function") throw new Error("physical_gpu_campaign_health_reader_is_invalid");
}

function normalizeTimeouts(
  value: Partial<PhysicalGpuCampaignTimeouts> | undefined,
): PhysicalGpuCampaignTimeouts {
  if (value !== undefined && (typeof value !== "object" || value === null || Array.isArray(value))) {
    throw new Error("physical_gpu_campaign_timeouts_must_be_an_object");
  }
  const candidate = (value ?? {}) as Record<string, unknown>;
  const allowed = new Set([
    "agentHealthMs",
    "apiHealthMs",
    "apiRequestMs",
    "sseIdleMs",
    "cleanupStopMs",
  ]);
  if (Object.keys(candidate).some((key) => !allowed.has(key))) {
    throw new Error("physical_gpu_campaign_timeouts_has_invalid_keys");
  }
  return {
    agentHealthMs: integer(
      candidate.agentHealthMs ?? DEFAULT_TIMEOUTS.agentHealthMs,
      1,
      MAX_TIMEOUT_MS,
      "physical_gpu_campaign_agent_health_timeout",
    ),
    apiHealthMs: integer(
      candidate.apiHealthMs ?? DEFAULT_TIMEOUTS.apiHealthMs,
      1,
      MAX_TIMEOUT_MS,
      "physical_gpu_campaign_api_health_timeout",
    ),
    apiRequestMs: integer(
      candidate.apiRequestMs ?? DEFAULT_TIMEOUTS.apiRequestMs,
      1,
      MAX_TIMEOUT_MS,
      "physical_gpu_campaign_api_request_timeout",
    ),
    sseIdleMs: integer(
      candidate.sseIdleMs ?? DEFAULT_TIMEOUTS.sseIdleMs,
      1,
      MAX_TIMEOUT_MS,
      "physical_gpu_campaign_sse_idle_timeout",
    ),
    cleanupStopMs: integer(
      candidate.cleanupStopMs ?? DEFAULT_TIMEOUTS.cleanupStopMs,
      1,
      MAX_TIMEOUT_MS,
      "physical_gpu_campaign_cleanup_stop_timeout",
    ),
  };
}

async function withCampaignDeadline<T>(
  operation: string,
  timeoutMs: number,
  task: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const controller = new AbortController();
  const timeoutError = new Error(`physical_gpu_campaign_timeout:${operation}:${timeoutMs}`);
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      if (!controller.signal.aborted) controller.abort(timeoutError);
      reject(timeoutError);
    }, timeoutMs);
  });
  const pending = Promise.resolve().then(() => task(controller.signal));
  try {
    return await Promise.race([pending, deadline]);
  } catch (error) {
    if (!controller.signal.aborted) controller.abort(error);
    throw normalizeError(error);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function campaignCancellation(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new Error("physical_gpu_campaign_cancelled");
}

function checkedClock(clock: () => number): () => number {
  if (typeof clock !== "function") throw new Error("physical_gpu_campaign_clock_is_invalid");
  return () => {
    const value = clock();
    if (!Number.isFinite(value) || value < 0) {
      throw new Error("physical_gpu_campaign_clock_returned_invalid_time");
    }
    return value;
  };
}

function duration(startedAt: number, endedAt: number, name: string): number {
  const value = endedAt - startedAt;
  if (!Number.isFinite(value) || value < 0) throw new Error(`${name}_is_invalid`);
  return value;
}

async function responseJsonObject(response: Response, name: string): Promise<Record<string, unknown>> {
  const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
  if (!contentType.includes("application/json")) throw new Error(`${name}_content_type_mismatch`);
  try {
    return object(await response.json(), name);
  } catch (error) {
    throw new Error(`${name}_is_invalid_json`, { cause: error });
  }
}

function jsonObject(value: string, name: string): Record<string, unknown> {
  try {
    return object(JSON.parse(value), name);
  } catch (error) {
    throw new Error(`${name}_is_invalid_json`, { cause: error });
  }
}

function firstChoice(document: Record<string, unknown>, name: string): Record<string, unknown> {
  const choices = array(document.choices, 1, 1, name);
  return object(choices[0], name);
}

function object(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${name}_must_be_an_object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, expected: string[], name: string): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (actual.length !== wanted.length || actual.some((key, index) => key !== wanted[index])) {
    throw new Error(`${name}_has_invalid_keys`);
  }
}

function array(value: unknown, minimum: number, maximum: number, name: string): unknown[] {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) {
    throw new Error(`${name}_has_invalid_length`);
  }
  return value;
}

function integer(value: unknown, minimum: number, maximum: number, name: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new Error(`${name}_must_be_an_integer`);
  }
  return value as number;
}

function integerArray(
  value: unknown,
  minimum: number,
  maximum: number,
  name: string,
): number[] {
  return array(value, minimum, maximum, name).map((item) =>
    integer(item, 0, Number.MAX_SAFE_INTEGER, name),
  );
}

function nonNegative(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`${name}_must_be_non_negative`);
  }
  return value;
}

function requiredText(value: unknown, name: string, maximum = 4_096): string {
  if (typeof value !== "string" || value.length < 1 || value.length > maximum) {
    throw new Error(`${name}_must_be_text`);
  }
  return value;
}

function identifier(value: unknown, name: string): string {
  const result = requiredText(value, name, 256);
  if (!/^[A-Za-z0-9][A-Za-z0-9._:@/-]*$/.test(result)) throw new Error(`${name}_is_invalid`);
  return result;
}

function sha256(value: unknown, name: string): string {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    throw new Error(`${name}_must_be_sha256`);
  }
  return value;
}

function decimalUint64(value: unknown, name: string): string {
  if (typeof value !== "string" || !DECIMAL_UINT64_PATTERN.test(value)) {
    throw new Error(`${name}_must_be_decimal_uint64`);
  }
  if (BigInt(value) > MAX_UINT64) throw new Error(`${name}_must_be_decimal_uint64`);
  return value;
}

function oneOf<const T extends readonly string[]>(value: unknown, expected: T, name: string): T[number] {
  if (typeof value !== "string" || !expected.includes(value)) throw new Error(`${name}_is_invalid`);
  return value as T[number];
}

function normalizeBaseUrl(value: unknown): string {
  const raw = requiredText(value, "physical_gpu_campaign_api_base_url");
  let endpoint: URL;
  try {
    endpoint = new URL(raw);
  } catch (error) {
    throw new Error("physical_gpu_campaign_api_base_url_is_invalid", { cause: error });
  }
  if (
    (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") ||
    endpoint.username !== "" ||
    endpoint.password !== "" ||
    endpoint.search !== "" ||
    endpoint.hash !== "" ||
    (endpoint.pathname !== "" && endpoint.pathname !== "/")
  ) {
    throw new Error("physical_gpu_campaign_api_base_url_is_invalid");
  }
  return endpoint.href.replace(/\/$/, "");
}

function assertUnique<T>(values: T[], message: string): void {
  if (new Set(values).size !== values.length) throw new Error(message);
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

function sameSet<T>(left: T[], right: T[]): boolean {
  return left.length === right.length && new Set(left).size === left.length && left.every((x) => right.includes(x));
}

function sameNumberArray(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function isNumber(value: number | null): value is number {
  return value !== null;
}

function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}
