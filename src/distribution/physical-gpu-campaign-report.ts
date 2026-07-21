import { canonicalEvidenceJson, sha256CanonicalEvidence } from "../core/json.js";
import {
  OUTPUT_TOKEN_HASH_SCHEME,
  PHYSICAL_GPU_CAMPAIGN_SCHEMA,
  type PhysicalGpuCampaignCanaryObservation,
  type PhysicalGpuCampaignObservation,
  type PhysicalGpuCampaignRequestSample,
} from "./physical-gpu-campaign.js";
import {
  OUTPUT_TOKEN_IDS_HASH_SCHEME,
  buildPhysicalTwoHostGpuGateReport,
  outputTokenIdsSha256,
  requirePassingPhysicalTwoHostGpuGate,
  type PhysicalGateDirectionalLinkEvidenceV1,
  type PhysicalGateRankWorkEvidenceV1,
  type PhysicalGateRequestSampleV1,
  type PhysicalTwoHostGpuGateReportV1,
} from "./physical-gpu-gate-report.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
} from "./python-launcher.js";
import {
  validatePhysicalProbe,
  type PhysicalProbeDeviceV1,
  type PhysicalProbeV1,
} from "./physical-probe.js";

export interface PhysicalGpuCampaignReportHostBinding {
  hostId: string;
  agentId: string;
  agentEndpoint: string;
  rankNodeId: string;
  device: string;
  offeredVramBytes: number;
  vendor: string;
  expectedProbeNonce: string;
  probe: PhysicalProbeV1;
}

export interface PhysicalGpuCampaignReportReference {
  referenceCanaryId: string;
  artifactIdentity: string;
  canonicalSource: string;
  canonicalRevision: string | null;
  tokenizerId: string;
  promptTokenIdsSha256: string;
  outputTokenIds: number[];
}

export interface PhysicalGpuCampaignReportInput {
  capturedAt: string;
  networkScope: "lan" | "wan";
  samplesTruncated: boolean;
  campaign: PhysicalGpuCampaignObservation;
  launch: PythonPipelineLaunchDescription;
  hosts: PhysicalGpuCampaignReportHostBinding[];
  networkLinks: PhysicalGateDirectionalLinkEvidenceV1[];
  reference: PhysicalGpuCampaignReportReference;
  rankWork: PhysicalGateRankWorkEvidenceV1[];
}

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const CUDA_DEVICE_PATTERN = /^cuda:(?:0|[1-9][0-9]*)$/;

/**
 * Pure, fail-closed adapter from one completed physical campaign to the
 * standalone sealed two-host GPU gate report.
 */
export function buildPhysicalGpuCampaignGateReport(
  inputValue: unknown,
): PhysicalTwoHostGpuGateReportV1 {
  canonicalEvidenceJson(inputValue);
  const input = validateInput(inputValue);
  validateCampaignConsistency(input.campaign, input.launch, input.hosts);
  if (input.samplesTruncated) {
    throw new Error("physical_gpu_campaign_report_samples_are_truncated");
  }

  const referenceDigest = outputTokenIdsSha256(input.reference.outputTokenIds);
  validateReferenceCanary(
    input.campaign.canaries.pre,
    "pre",
    input.reference.referenceCanaryId,
    input.reference.outputTokenIds.length,
    referenceDigest,
  );
  validateReferenceCanary(
    input.campaign.canaries.post,
    "post",
    input.reference.referenceCanaryId,
    input.reference.outputTokenIds.length,
    referenceDigest,
  );

  const selectedSamples = input.campaign.samples.filter(
    (sample) => sample.canaryId === input.reference.referenceCanaryId,
  );
  const mappedSamples = selectedSamples.map((sample) =>
    mapSample(sample, input.reference.outputTokenIds.length, referenceDigest),
  );
  const warmups = mappedSamples.filter((sample) => sample.phase === "warmup");
  const measured = mappedSamples.filter((sample) => sample.phase === "measure");
  if (warmups.length < 1) {
    throw new Error("physical_gpu_campaign_report_requires_reference_warmup");
  }
  if (measured.length < 5) {
    throw new Error("physical_gpu_campaign_report_requires_five_reference_measurements");
  }
  if (measured.some((sample) => sample.completionTokens < 16)) {
    throw new Error("physical_gpu_campaign_report_reference_measurement_is_too_short");
  }

  const hosts = input.hosts.map(mapHost);
  const started = input.campaign.lifecycle.supervisorStarted!;
  const stopped = input.campaign.lifecycle.supervisorStopped!;
  const health = input.campaign.apiHealth!;
  const report = buildPhysicalTwoHostGpuGateReport({
    capturedAt: input.capturedAt,
    provenance: {
      level: "hardware-physical",
      source: "measurement",
      networkScope: input.networkScope,
      loopback: false,
      emulated: false,
      attestation: "self-reported",
    },
    launch: {
      description: structuredClone(input.launch),
      canonicalSha256: sha256CanonicalEvidence(input.launch),
    },
    hosts,
    networkLinks: structuredClone(input.networkLinks),
    lifecycle: {
      readyProcessIds: started.processes.map((process) => process.processId),
      stoppedProcessIds: stopped.processes.map((process) => process.processId),
      residualProcessIds: [],
      health: {
        status: health.status,
        model: health.model,
        stages: health.stages,
        boundaries: [...health.boundaries],
        codec: health.codec,
      },
    },
    reference: {
      mode: "monolithic-greedy",
      modelId: input.launch.modelIdentity.id,
      modelRevision: input.launch.modelIdentity.revision,
      // This value came from canary_reference.py and has already been bound
      // exactly to the sealed launch below; it is not inferred from READY or
      // copied from the launch without reference-generator evidence.
      tokenizerId: input.reference.tokenizerId,
      promptTokenIdsSha256: input.reference.promptTokenIdsSha256,
      outputTokenIds: [...input.reference.outputTokenIds],
      outputTokenIdsHashScheme: OUTPUT_TOKEN_IDS_HASH_SCHEME,
      outputTokenIdsSha256: referenceDigest,
    },
    rankWork: structuredClone(input.rankWork),
    samples: mappedSamples,
  });
  return requirePassingPhysicalTwoHostGpuGate(report);
}

function validateInput(value: unknown): PhysicalGpuCampaignReportInput {
  const input = record(value, "physical_gpu_campaign_report_input");
  exactKeys(
    input,
    [
      "capturedAt",
      "networkScope",
      "samplesTruncated",
      "campaign",
      "launch",
      "hosts",
      "networkLinks",
      "reference",
      "rankWork",
    ],
    "physical_gpu_campaign_report_input",
  );
  isoTimestamp(input.capturedAt, "physical_gpu_campaign_report_captured_at");
  if (input.networkScope !== "lan" && input.networkScope !== "wan") {
    throw new Error("physical_gpu_campaign_report_network_scope_is_invalid");
  }
  if (typeof input.samplesTruncated !== "boolean") {
    throw new Error("physical_gpu_campaign_report_truncation_flag_is_invalid");
  }
  try {
    validatePythonLaunchDescription(input.launch);
  } catch (error) {
    throw new Error("physical_gpu_campaign_report_launch_is_invalid", { cause: error });
  }
  const hosts = array(input.hosts, 2, 2, "physical_gpu_campaign_report_hosts").map(
    validateHostBinding,
  );
  const nonces = hosts.map((host) => host.expectedProbeNonce);
  if (new Set(nonces).size !== nonces.length) {
    throw new Error("physical_gpu_campaign_report_probe_nonces_are_not_unique");
  }
  const hostFingerprints = hosts.map((host) => host.probe.host.fingerprintSha256);
  if (new Set(hostFingerprints).size !== hosts.length) {
    throw new Error("physical_gpu_campaign_report_probe_hosts_are_not_unique");
  }
  const deviceFingerprints = hosts.map((host) => selectedDevice(host).fingerprintSha256);
  if (new Set(deviceFingerprints).size !== hosts.length) {
    throw new Error("physical_gpu_campaign_report_probe_gpus_are_not_unique");
  }
  const links = array(
    input.networkLinks,
    2,
    2,
    "physical_gpu_campaign_report_network_links",
  ) as PhysicalGateDirectionalLinkEvidenceV1[];
  const rankWork = array(
    input.rankWork,
    2,
    2,
    "physical_gpu_campaign_report_rank_work",
  ) as PhysicalGateRankWorkEvidenceV1[];
  const reference = validateReference(input.reference);
  validateReferenceIdentity(reference, input.launch);
  return {
    capturedAt: input.capturedAt,
    networkScope: input.networkScope,
    samplesTruncated: input.samplesTruncated,
    campaign: input.campaign as PhysicalGpuCampaignObservation,
    launch: structuredClone(input.launch),
    hosts,
    networkLinks: structuredClone(links),
    reference,
    rankWork: structuredClone(rankWork),
  };
}

function validateHostBinding(
  value: unknown,
  index: number,
): PhysicalGpuCampaignReportHostBinding {
  const host = record(value, `physical_gpu_campaign_report_host_${index}`);
  exactKeys(
    host,
    [
      "hostId",
      "agentId",
      "agentEndpoint",
      "rankNodeId",
      "device",
      "offeredVramBytes",
      "vendor",
      "expectedProbeNonce",
      "probe",
    ],
    `physical_gpu_campaign_report_host_${index}`,
  );
  for (const [field, name] of [
    [host.hostId, "host_id"],
    [host.agentId, "agent_id"],
    [host.agentEndpoint, "agent_endpoint"],
    [host.rankNodeId, "rank_node_id"],
    [host.device, "device"],
    [host.vendor, "vendor"],
    [host.expectedProbeNonce, "probe_nonce"],
  ] as const) {
    text(field, `physical_gpu_campaign_report_${name}_is_invalid`);
  }
  if (!CUDA_DEVICE_PATTERN.test(host.device)) {
    throw new Error("physical_gpu_campaign_report_device_is_invalid");
  }
  integer(
    host.offeredVramBytes,
    1,
    Number.MAX_SAFE_INTEGER,
    "physical_gpu_campaign_report_offered_vram_is_invalid",
  );
  try {
    validatePhysicalProbe(host.probe, host.expectedProbeNonce);
  } catch (error) {
    throw new Error("physical_gpu_campaign_report_probe_is_invalid", { cause: error });
  }
  const result = host as unknown as PhysicalGpuCampaignReportHostBinding;
  const device = selectedDevice(result);
  if (result.offeredVramBytes > device.totalMemoryBytes) {
    throw new Error("physical_gpu_campaign_report_offered_vram_exceeds_device");
  }
  if (
    !result.probe.runtime.cudaApiAvailable ||
    !result.probe.runtime.distributedAvailable ||
    !result.probe.runtime.ncclAvailable
  ) {
    throw new Error("physical_gpu_campaign_report_probe_is_not_gpu_collective_capable");
  }
  return structuredClone(result);
}

function validateReference(value: unknown): PhysicalGpuCampaignReportReference {
  const reference = record(value, "physical_gpu_campaign_report_reference");
  exactKeys(
    reference,
    [
      "referenceCanaryId",
      "artifactIdentity",
      "canonicalSource",
      "canonicalRevision",
      "tokenizerId",
      "promptTokenIdsSha256",
      "outputTokenIds",
    ],
    "physical_gpu_campaign_report_reference",
  );
  text(reference.referenceCanaryId, "physical_gpu_campaign_report_reference_canary_is_invalid");
  sha256(
    reference.artifactIdentity,
    "physical_gpu_campaign_report_reference_artifact_identity_is_invalid",
  );
  text(
    reference.canonicalSource,
    "physical_gpu_campaign_report_reference_canonical_source_is_invalid",
  );
  if (reference.canonicalRevision !== null) {
    text(
      reference.canonicalRevision,
      "physical_gpu_campaign_report_reference_canonical_revision_is_invalid",
    );
  }
  text(
    reference.tokenizerId,
    "physical_gpu_campaign_report_reference_tokenizer_is_invalid",
  );
  sha256(
    reference.promptTokenIdsSha256,
    "physical_gpu_campaign_report_prompt_hash_is_invalid",
  );
  const outputTokenIds = array(
    reference.outputTokenIds,
    1,
    32_768,
    "physical_gpu_campaign_report_reference_tokens",
  ).map((tokenId) =>
    integer(
      tokenId,
      0,
      0xffff_ffff,
      "physical_gpu_campaign_report_reference_token_is_invalid",
    ),
  );
  return {
    referenceCanaryId: reference.referenceCanaryId,
    artifactIdentity: reference.artifactIdentity,
    canonicalSource: reference.canonicalSource,
    canonicalRevision: reference.canonicalRevision,
    tokenizerId: reference.tokenizerId,
    promptTokenIdsSha256: reference.promptTokenIdsSha256,
    outputTokenIds,
  };
}

function validateReferenceIdentity(
  reference: PhysicalGpuCampaignReportReference,
  launch: PythonPipelineLaunchDescription,
): void {
  if (reference.artifactIdentity !== launch.runtimeModel.artifactIdentity) {
    throw new Error("physical_gpu_campaign_report_reference_artifact_identity_mismatch");
  }
  if (reference.canonicalSource !== launch.runtimeModel.canonicalSource) {
    throw new Error("physical_gpu_campaign_report_reference_canonical_source_mismatch");
  }
  if (reference.canonicalRevision !== launch.runtimeModel.canonicalRevision) {
    throw new Error("physical_gpu_campaign_report_reference_canonical_revision_mismatch");
  }
  if (reference.tokenizerId !== launch.modelIdentity.tokenizerId) {
    throw new Error("physical_gpu_campaign_report_reference_tokenizer_mismatch");
  }
}

function validateCampaignConsistency(
  campaignValue: unknown,
  launch: PythonPipelineLaunchDescription,
  hosts: PhysicalGpuCampaignReportHostBinding[],
): asserts campaignValue is PhysicalGpuCampaignObservation {
  const campaign = record(campaignValue, "physical_gpu_campaign_report_campaign");
  if (campaign.schema !== PHYSICAL_GPU_CAMPAIGN_SCHEMA) {
    throw new Error("physical_gpu_campaign_report_campaign_schema_is_invalid");
  }
  if (campaign.passed !== true) {
    throw new Error("physical_gpu_campaign_report_campaign_did_not_pass");
  }
  if (campaign.launchId !== launch.launchId || campaign.pipelineId !== launch.pipelineId) {
    throw new Error("physical_gpu_campaign_report_campaign_launch_mismatch");
  }
  if (!Array.isArray(campaign.failures) || campaign.failures.length !== 0) {
    throw new Error("physical_gpu_campaign_report_campaign_has_failures");
  }
  if (!Array.isArray(campaign.samples) || !Array.isArray(campaign.batches)) {
    throw new Error("physical_gpu_campaign_report_campaign_samples_are_invalid");
  }
  if (
    campaign.samples.some((sample) => !recordOrFalse(sample)) ||
    campaign.batches.some((batch) => !recordOrFalse(batch))
  ) {
    throw new Error("physical_gpu_campaign_report_campaign_samples_are_invalid");
  }
  const sampleIds = (campaign.samples as PhysicalGpuCampaignRequestSample[]).map(
    (sample) => sample.sampleId,
  );
  if (new Set(sampleIds).size !== sampleIds.length) {
    throw new Error("physical_gpu_campaign_report_campaign_sample_ids_are_not_unique");
  }
  if ((campaign.batches as Array<{ passed?: unknown }>).some((batch) => batch.passed !== true)) {
    throw new Error("physical_gpu_campaign_report_campaign_has_failed_batches");
  }
  const campaignSamples = campaign.samples as PhysicalGpuCampaignRequestSample[];
  if (campaignSamples.some((sample) => sample.passed !== true || sample.error !== null)) {
    throw new Error("physical_gpu_campaign_report_campaign_has_failed_samples");
  }
  const measuredSamples = campaignSamples.filter((sample) => sample.phase === "measure");
  const measuredTokens = measuredSamples.reduce(
    (total, sample) => total + (sample.completionTokens ?? 0),
    0,
  );
  const summary = record(campaign.summary, "physical_gpu_campaign_report_campaign_summary");
  if (
    summary.measuredRequests !== measuredSamples.length ||
    summary.actualCompletionTokens !== measuredTokens
  ) {
    throw new Error("physical_gpu_campaign_report_campaign_summary_mismatch");
  }
  const lifecycle = record(campaign.lifecycle, "physical_gpu_campaign_report_lifecycle");
  if (lifecycle.cleanupAttempted !== true || lifecycle.cleanupPassed !== true) {
    throw new Error("physical_gpu_campaign_report_campaign_cleanup_failed");
  }
  const started = requireSupervisorSnapshot(lifecycle.supervisorStarted, "running", "ready", launch);
  requireSupervisorSnapshot(lifecycle.supervisorBeforeStop, "running", "ready", launch);
  const stopped = requireSupervisorSnapshot(lifecycle.supervisorStopped, "stopped", "stopped", launch);
  if (started.processes.length !== stopped.processes.length) {
    throw new Error("physical_gpu_campaign_report_supervisor_process_count_changed");
  }
  validateAgentCleanup(lifecycle.agentHealthBefore, lifecycle.agentHealthAfter, launch, hosts);

  const health = record(campaign.apiHealth, "physical_gpu_campaign_report_api_health");
  if (
    health.status !== "ready" ||
    health.error !== null ||
    health.model !== launch.configuration.publicModelName ||
    health.artifactIdentity !== launch.runtimeModel.artifactIdentity ||
    health.canonicalModelSource !== launch.runtimeModel.canonicalSource ||
    health.canonicalModelRevision !== (launch.runtimeModel.canonicalRevision ?? null) ||
    health.pipelineSnapshotIdentity !== launch.runtimeModel.snapshotIdentity
  ) {
    throw new Error("physical_gpu_campaign_report_api_health_mismatch");
  }
  const root = launch.launchOrder.find((process) => process.kind === "root-engine");
  if (
    root?.kind !== "root-engine" ||
    health.stages !== launch.route.logicalStageIds.length ||
    health.codec !== launch.route.codec ||
    canonicalEvidenceJson(health.boundaries) !== canonicalEvidenceJson(root.boundaries)
  ) {
    throw new Error("physical_gpu_campaign_report_api_topology_mismatch");
  }
  const canaries = record(campaign.canaries, "physical_gpu_campaign_report_canaries");
  if (!Array.isArray(canaries.pre) || !Array.isArray(canaries.post)) {
    throw new Error("physical_gpu_campaign_report_canaries_are_invalid");
  }
  for (const observation of [...canaries.pre, ...canaries.post]) {
    if (
      !recordOrFalse(observation) ||
      observation.passed !== true ||
      observation.error !== null ||
      !recordOrFalse(observation.evidence)
    ) {
      throw new Error("physical_gpu_campaign_report_campaign_has_failed_canaries");
    }
  }
}

function validateReferenceCanary(
  observations: PhysicalGpuCampaignCanaryObservation[],
  phase: "pre" | "post",
  canaryId: string,
  expectedCount: number,
  expectedDigest: string,
): void {
  const matching = observations.filter(
    (observation) => observation.phase === phase && observation.canaryId === canaryId,
  );
  if (matching.length !== 1) {
    throw new Error(`physical_gpu_campaign_report_${phase}_reference_canary_is_missing`);
  }
  const observation = matching[0]!;
  const evidence = observation.evidence;
  if (
    !observation.passed ||
    observation.error !== null ||
    evidence === null ||
    evidence.outputTokenIdsHashScheme !== OUTPUT_TOKEN_HASH_SCHEME ||
    evidence.outputTokenIdsSha256 !== expectedDigest ||
    evidence.completionTokens !== expectedCount ||
    !positiveFinite(observation.clientResponseMs)
  ) {
    throw new Error(`physical_gpu_campaign_report_${phase}_reference_canary_mismatch`);
  }
}

function mapSample(
  sample: PhysicalGpuCampaignRequestSample,
  expectedCount: number,
  expectedDigest: string,
): PhysicalGateRequestSampleV1 {
  if (
    !sample.passed ||
    sample.error !== null ||
    (sample.finishReason !== "length" && sample.finishReason !== "stop") ||
    sample.completionTokens !== expectedCount ||
    sample.outputTokenIdsSha256 !== expectedDigest ||
    !nonNegativeFinite(sample.clientFirstContentMs) ||
    !positiveFinite(sample.clientResponseMs) ||
    sample.clientFirstContentMs > sample.clientResponseMs ||
    !positiveInteger(sample.promptTokens) ||
    !nonNegativeFinite(sample.serverTtftMs) ||
    !nonNegativeFinite(sample.serverTpotMs) ||
    !positiveFinite(sample.serverPipelineMs) ||
    !positiveFinite(sample.perUserOutputTokensPerSecondIncludingTtft)
  ) {
    throw new Error(`physical_gpu_campaign_report_sample_is_invalid:${sample.sampleId}`);
  }
  const reconstructed =
    sample.serverTtftMs + Math.max(0, sample.completionTokens - 1) * sample.serverTpotMs;
  const tolerance = Math.max(0.5, sample.serverPipelineMs * 0.005);
  if (Math.abs(sample.serverPipelineMs - reconstructed) > tolerance) {
    throw new Error(`physical_gpu_campaign_report_sample_metrics_are_inconsistent:${sample.sampleId}`);
  }
  const expectedRate = sample.completionTokens / (sample.clientResponseMs / 1_000);
  if (!nearlyEqual(sample.perUserOutputTokensPerSecondIncludingTtft, expectedRate)) {
    throw new Error(`physical_gpu_campaign_report_sample_rate_is_inconsistent:${sample.sampleId}`);
  }
  return {
    sampleId: sample.sampleId,
    phase: sample.phase,
    concurrency: sample.concurrency,
    iteration: sample.iteration,
    promptTokens: sample.promptTokens,
    completionTokens: sample.completionTokens,
    outputTokenIdsHashScheme: OUTPUT_TOKEN_IDS_HASH_SCHEME,
    outputTokenIdsSha256: sample.outputTokenIdsSha256,
    // Externally observed first content is the user-visible TTFT. TPOT and
    // pipeline time remain the server's token-level measurements.
    ttftMs: sample.clientFirstContentMs,
    tpotMs: sample.serverTpotMs,
    responseMs: sample.clientResponseMs,
    pipelineMs: sample.serverPipelineMs,
  };
}

function mapHost(binding: PhysicalGpuCampaignReportHostBinding) {
  const device = selectedDevice(binding);
  return {
    hostId: binding.hostId,
    hostFingerprintSha256: binding.probe.host.fingerprintSha256,
    agentId: binding.agentId,
    agentEndpoint: binding.agentEndpoint,
    rankNodeId: binding.rankNodeId,
    gpu: {
      deviceFingerprintSha256: device.fingerprintSha256,
      device: binding.device,
      vendor: binding.vendor,
      model: device.name,
      physicalVramBytes: device.totalMemoryBytes,
      offeredVramBytes: binding.offeredVramBytes,
      computeApi: binding.probe.runtime.rocmVersion === null ? ("cuda" as const) : ("rocm" as const),
      runtimeAvailable: binding.probe.runtime.cudaApiAvailable,
      collectiveAvailable:
        binding.probe.runtime.distributedAvailable && binding.probe.runtime.ncclAvailable,
    },
  };
}

function selectedDevice(binding: PhysicalGpuCampaignReportHostBinding): PhysicalProbeDeviceV1 {
  const match = CUDA_DEVICE_PATTERN.exec(binding.device);
  if (!match) throw new Error("physical_gpu_campaign_report_device_is_invalid");
  const index = Number(binding.device.slice("cuda:".length));
  const device = binding.probe.devices.find((candidate) => candidate.index === index);
  if (device === undefined) {
    throw new Error("physical_gpu_campaign_report_device_is_not_in_probe");
  }
  return device;
}

function requireSupervisorSnapshot(
  value: unknown,
  expectedState: "running" | "stopped",
  expectedProcessState: "ready" | "stopped",
  launch: PythonPipelineLaunchDescription,
) {
  const snapshot = record(value, "physical_gpu_campaign_report_supervisor_snapshot");
  if (snapshot.state !== expectedState || snapshot.failure !== null || !Array.isArray(snapshot.processes)) {
    throw new Error("physical_gpu_campaign_report_supervisor_state_is_invalid");
  }
  const expected = [...launch.launchOrder.map((process) => process.processId)].sort();
  const processes = snapshot.processes as Array<{ processId?: unknown; state?: unknown }>;
  const observed = processes.map((process) => process.processId).sort();
  if (
    canonicalEvidenceJson(observed) !== canonicalEvidenceJson(expected) ||
    processes.some((process) => process.state !== expectedProcessState)
  ) {
    throw new Error("physical_gpu_campaign_report_supervisor_processes_are_invalid");
  }
  return snapshot as unknown as NonNullable<PhysicalGpuCampaignObservation["lifecycle"]["supervisorStarted"]>;
}

function validateAgentCleanup(
  beforeValue: unknown,
  afterValue: unknown,
  launch: PythonPipelineLaunchDescription,
  hosts: PhysicalGpuCampaignReportHostBinding[],
): void {
  if (!Array.isArray(beforeValue) || !Array.isArray(afterValue)) {
    throw new Error("physical_gpu_campaign_report_agent_health_is_invalid");
  }
  const expectedNodes = new Set(launch.launchOrder.map((process) => process.anchor.memberId));
  for (const observations of [beforeValue, afterValue]) {
    if (observations.length !== expectedNodes.size) {
      throw new Error("physical_gpu_campaign_report_agent_health_count_mismatch");
    }
    const observedNodes = new Set<string>();
    for (const value of observations) {
      const observation = record(value, "physical_gpu_campaign_report_agent_health_item");
      const health = record(observation.health, "physical_gpu_campaign_report_agent_health_value");
      if (
        observation.passed !== true ||
        observation.error !== null ||
        typeof observation.expectedNodeId !== "string" ||
        typeof observation.expectedAgentId !== "string" ||
        health.agentId !== observation.expectedAgentId ||
        health.nodeId !== observation.expectedNodeId ||
        health.processes !== 0 ||
        !expectedNodes.has(observation.expectedNodeId)
      ) {
        throw new Error("physical_gpu_campaign_report_agent_health_mismatch");
      }
      observedNodes.add(observation.expectedNodeId);
    }
    if (observedNodes.size !== expectedNodes.size) {
      throw new Error("physical_gpu_campaign_report_agent_health_nodes_are_not_unique");
    }
  }
  for (const host of hosts) {
    const matching = afterValue.filter(
      (value) =>
        recordOrFalse(value) &&
        value.expectedNodeId === host.rankNodeId &&
        value.expectedAgentId === host.agentId,
    );
    if (matching.length !== 1) {
      throw new Error("physical_gpu_campaign_report_host_agent_binding_mismatch");
    }
  }
}

function record(value: unknown, name: string): Record<string, any> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name}_is_invalid`);
  }
  return value as Record<string, any>;
}

function recordOrFalse(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
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

function text(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !value.trim() || /[\0\r\n]/.test(value)) {
    throw new Error(name);
  }
}

function integer(
  value: unknown,
  minimum: number,
  maximum: number,
  name: string,
): number {
  if (!Number.isSafeInteger(value) || (value as number) < minimum || (value as number) > maximum) {
    throw new Error(name);
  }
  return value as number;
}

function sha256(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) throw new Error(name);
}

function isoTimestamp(value: unknown, name: string): asserts value is string {
  if (
    typeof value !== "string" ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value) ||
    new Date(value).toISOString() !== value
  ) {
    throw new Error(name);
  }
}

function nonNegativeFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function positiveFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function positiveInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) > 0;
}

function nearlyEqual(left: number, right: number): boolean {
  return Math.abs(left - right) <= Math.max(1e-9, Math.abs(right) * 1e-9);
}
