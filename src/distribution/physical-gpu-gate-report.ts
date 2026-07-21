import { createHash } from "node:crypto";
import {
  canonicalEvidenceJson,
  sha256CanonicalEvidence,
} from "../core/json.js";
import {
  validatePythonLaunchDescription,
  type PythonPipelineLaunchDescription,
  type PythonRemoteStageLaunch,
} from "./python-launcher.js";

export const PHYSICAL_TWO_HOST_GPU_GATE_SCHEMA =
  "gdlp-physical-two-host-gpu-gate/1" as const;
export const PHYSICAL_TWO_HOST_GPU_GATE_POLICY_SCHEMA =
  "gdlp-physical-two-host-gpu-policy/1" as const;
export const PHYSICAL_GPU_GATE_CANONICALIZATION =
  "gdlp-canonical-evidence-json/1" as const;
export const OUTPUT_TOKEN_IDS_HASH_SCHEME = "gdlp-output-token-ids-v1" as const;

export interface PhysicalTwoHostGpuGatePolicyV1 {
  schema: typeof PHYSICAL_TWO_HOST_GPU_GATE_POLICY_SCHEMA;
  minimumWarmupSamples: number;
  minimumMeasuredSamples: number;
  minimumCompletionTokensPerMeasuredSample: number;
  minimumDirectionalLinkSamples: number;
}

export const PHYSICAL_TWO_HOST_GPU_GATE_POLICY: Readonly<PhysicalTwoHostGpuGatePolicyV1> =
  Object.freeze({
    schema: PHYSICAL_TWO_HOST_GPU_GATE_POLICY_SCHEMA,
    minimumWarmupSamples: 1,
    minimumMeasuredSamples: 5,
    minimumCompletionTokensPerMeasuredSample: 16,
    minimumDirectionalLinkSamples: 3,
  });

export type PhysicalEvidenceSource = "measurement" | "projection";
export type PhysicalNetworkScope = "lan" | "wan" | "loopback";
export type PhysicalGpuComputeApi = "cuda" | "rocm" | "cpu";
export type PhysicalSamplePhase = "warmup" | "measure";

export interface PhysicalGateProvenanceV1 {
  level: "hardware-physical";
  source: PhysicalEvidenceSource;
  networkScope: PhysicalNetworkScope;
  loopback: boolean;
  emulated: boolean;
  attestation: "self-reported";
}

export interface PhysicalGateLaunchEvidenceV1 {
  description: PythonPipelineLaunchDescription;
  canonicalSha256: string;
}

export interface PhysicalGateGpuEvidenceV1 {
  deviceFingerprintSha256: string;
  device: string;
  vendor: string;
  model: string;
  physicalVramBytes: number;
  offeredVramBytes: number;
  computeApi: PhysicalGpuComputeApi;
  runtimeAvailable: boolean;
  collectiveAvailable: boolean;
}

export interface PhysicalGateHostEvidenceV1 {
  hostId: string;
  hostFingerprintSha256: string;
  agentId: string;
  agentEndpoint: string;
  rankNodeId: string;
  gpu: PhysicalGateGpuEvidenceV1;
}

export interface PhysicalGateDirectionalLinkEvidenceV1 {
  fromHostId: string;
  toHostId: string;
  rttMs: number[];
  goodputMbps: number[];
}

export interface PhysicalGateHealthEvidenceV1 {
  status: string;
  model: string;
  stages: number;
  boundaries: number[];
  codec: string;
}

export interface PhysicalGateLifecycleEvidenceV1 {
  readyProcessIds: string[];
  stoppedProcessIds: string[];
  residualProcessIds: string[];
  health: PhysicalGateHealthEvidenceV1;
}

export interface PhysicalGateReferenceEvidenceV1 {
  mode: "monolithic-greedy";
  modelId: string;
  modelRevision: string;
  tokenizerId: string;
  promptTokenIdsSha256: string;
  outputTokenIds: number[];
  outputTokenIdsHashScheme: typeof OUTPUT_TOKEN_IDS_HASH_SCHEME;
  outputTokenIdsSha256: string;
}

export interface PhysicalGateRankWorkEvidenceV1 {
  rank: number;
  hostId: string;
  nodeId: string;
  device: string;
  forwardCalls: number;
  collectiveCalls: number;
  tokensProcessed: number;
  bytesSent: number;
  bytesReceived: number;
  peakAllocatedBytes: number;
}

export interface PhysicalGateRequestSampleV1 {
  sampleId: string;
  phase: PhysicalSamplePhase;
  concurrency: number;
  iteration: number;
  promptTokens: number;
  completionTokens: number;
  outputTokenIdsHashScheme: typeof OUTPUT_TOKEN_IDS_HASH_SCHEME;
  outputTokenIdsSha256: string;
  ttftMs: number;
  tpotMs: number;
  responseMs: number;
  pipelineMs: number;
}

export interface PhysicalGateMetricStatsV1 {
  count: number;
  mean: number | null;
  p50: number | null;
  p95: number | null;
  min: number | null;
  max: number | null;
}

export interface PhysicalGateMetricSummaryV1 {
  warmupSamples: number;
  measuredSamples: number;
  measuredCompletionTokens: number;
  ttftMs: PhysicalGateMetricStatsV1;
  tpotMs: PhysicalGateMetricStatsV1;
  responseMs: PhysicalGateMetricStatsV1;
  pipelineMs: PhysicalGateMetricStatsV1;
  outputTokensPerSecondIncludingTtft: PhysicalGateMetricStatsV1;
}

export type PhysicalGateCheckId =
  | "physical_measurement"
  | "sealed_launch"
  | "two_distinct_hosts"
  | "two_distinct_gpus"
  | "non_loopback_route"
  | "two_rank_gpu_collective"
  | "one_rank_per_host"
  | "both_ranks_executed"
  | "directional_network_measured"
  | "exact_token_parity"
  | "sufficient_samples"
  | "healthy_clean_lifecycle";

export interface PhysicalGateCheckV1 {
  id: PhysicalGateCheckId;
  passed: boolean;
  detail: string;
}

export interface PhysicalTwoHostGpuGateDecisionV1 {
  passed: boolean;
  checks: PhysicalGateCheckV1[];
}

export interface PhysicalGpuGateSealV1 {
  canonicalization: typeof PHYSICAL_GPU_GATE_CANONICALIZATION;
  algorithm: "sha256";
  digest: string;
}

export interface PhysicalTwoHostGpuGateEvidenceV1 {
  capturedAt: string;
  provenance: PhysicalGateProvenanceV1;
  launch: PhysicalGateLaunchEvidenceV1;
  hosts: PhysicalGateHostEvidenceV1[];
  networkLinks: PhysicalGateDirectionalLinkEvidenceV1[];
  lifecycle: PhysicalGateLifecycleEvidenceV1;
  reference: PhysicalGateReferenceEvidenceV1;
  rankWork: PhysicalGateRankWorkEvidenceV1[];
  samples: PhysicalGateRequestSampleV1[];
}

export interface PhysicalTwoHostGpuGateReportV1
  extends PhysicalTwoHostGpuGateEvidenceV1 {
  schema: typeof PHYSICAL_TWO_HOST_GPU_GATE_SCHEMA;
  policy: PhysicalTwoHostGpuGatePolicyV1;
  summary: PhysicalGateMetricSummaryV1;
  gate: PhysicalTwoHostGpuGateDecisionV1;
  seal: PhysicalGpuGateSealV1;
}

type PhysicalGateReportBodyV1 = Omit<PhysicalTwoHostGpuGateReportV1, "seal">;

const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const MAX_HOSTS = 16;
const MAX_LINKS = 256;
const MAX_RANKS = 256;
const MAX_SAMPLES = 100_000;
const MAX_TOKEN_IDS = 32_768;

/**
 * Byte-for-byte counterpart of distributed_runtime.server.output_token_ids_sha256.
 * The count is uint64 big-endian and every token id is uint32 big-endian.
 */
export function outputTokenIdsSha256(tokenIds: readonly number[]): string {
  if (!Array.isArray(tokenIds) || tokenIds.length > MAX_TOKEN_IDS) {
    throw new Error("output_token_ids_must_be_a_bounded_array");
  }
  const digest = createHash("sha256");
  digest.update(`${OUTPUT_TOKEN_IDS_HASH_SCHEME}\0`, "ascii");
  const count = Buffer.allocUnsafe(8);
  count.writeBigUInt64BE(BigInt(tokenIds.length));
  digest.update(count);
  const encoded = Buffer.allocUnsafe(4);
  for (const tokenId of tokenIds) {
    if (!Number.isSafeInteger(tokenId) || tokenId < 0 || tokenId > 0xffff_ffff) {
      throw new Error("output_token_id_must_be_uint32");
    }
    encoded.writeUInt32BE(tokenId, 0);
    digest.update(encoded);
  }
  return `sha256:${digest.digest("hex")}`;
}

export function buildPhysicalTwoHostGpuGateReport(
  evidenceValue: PhysicalTwoHostGpuGateEvidenceV1,
): PhysicalTwoHostGpuGateReportV1 {
  canonicalEvidenceJson(evidenceValue);
  assertEvidence(evidenceValue, true);
  validateLaunch(evidenceValue.launch.description);

  const evidence = structuredClone(evidenceValue);
  const bodyWithoutDerived = {
    schema: PHYSICAL_TWO_HOST_GPU_GATE_SCHEMA,
    capturedAt: evidence.capturedAt,
    provenance: evidence.provenance,
    policy: structuredClone(PHYSICAL_TWO_HOST_GPU_GATE_POLICY),
    launch: evidence.launch,
    hosts: evidence.hosts,
    networkLinks: evidence.networkLinks,
    lifecycle: evidence.lifecycle,
    reference: evidence.reference,
    rankWork: evidence.rankWork,
    samples: evidence.samples,
  };
  const summary = summarizeSamples(bodyWithoutDerived.samples);
  const gate = deriveGate({ ...bodyWithoutDerived, summary });
  const body: PhysicalGateReportBodyV1 = { ...bodyWithoutDerived, summary, gate };
  const report: PhysicalTwoHostGpuGateReportV1 = {
    ...body,
    seal: {
      canonicalization: PHYSICAL_GPU_GATE_CANONICALIZATION,
      algorithm: "sha256",
      digest: sha256CanonicalEvidence(body),
    },
  };
  validatePhysicalTwoHostGpuGateReport(report);
  return report;
}

export function readPhysicalTwoHostGpuGateReport(
  source: string | unknown,
): PhysicalTwoHostGpuGateReportV1 {
  let value = source;
  if (typeof source === "string") {
    try {
      value = JSON.parse(source) as unknown;
    } catch {
      throw new Error("physical_gpu_gate_invalid_json");
    }
  }
  validatePhysicalTwoHostGpuGateReport(value);
  return value;
}

export function validatePhysicalTwoHostGpuGateReport(
  value: unknown,
): asserts value is PhysicalTwoHostGpuGateReportV1 {
  canonicalEvidenceJson(value);
  assertReport(value);
  validateLaunch(value.launch.description);

  const { seal, ...body } = value;
  const expectedDigest = sha256CanonicalEvidence(body);
  if (seal.digest !== expectedDigest) {
    throw new Error("physical_gpu_gate_seal_mismatch");
  }

  const expectedSummary = summarizeSamples(value.samples);
  if (canonicalEvidenceJson(value.summary) !== canonicalEvidenceJson(expectedSummary)) {
    throw new Error("physical_gpu_gate_summary_mismatch");
  }
  const expectedGate = deriveGate({ ...body, summary: expectedSummary });
  if (canonicalEvidenceJson(value.gate) !== canonicalEvidenceJson(expectedGate)) {
    throw new Error("physical_gpu_gate_decision_mismatch");
  }
}

export function evaluatePhysicalTwoHostGpuGate(
  value: unknown,
): PhysicalTwoHostGpuGateDecisionV1 {
  validatePhysicalTwoHostGpuGateReport(value);
  const { seal: _seal, ...body } = value;
  return deriveGate(body);
}

export function requirePassingPhysicalTwoHostGpuGate(
  source: string | unknown,
): PhysicalTwoHostGpuGateReportV1 {
  const report = readPhysicalTwoHostGpuGateReport(source);
  if (!report.gate.passed) {
    const failed = report.gate.checks
      .filter((check) => !check.passed)
      .map((check) => check.id)
      .join(",");
    throw new Error(`physical_gpu_gate_failed:${failed}`);
  }
  return report;
}

function deriveGate(
  body: Omit<PhysicalGateReportBodyV1, "gate">,
): PhysicalTwoHostGpuGateDecisionV1 {
  const hosts = body.hosts;
  const hostIds = hosts.map((host) => host.hostId);
  const hostFingerprints = hosts.map((host) => host.hostFingerprintSha256);
  const gpuFingerprints = hosts.map((host) => host.gpu.deviceFingerprintSha256);
  const launchHash = sha256CanonicalEvidence(body.launch.description);
  const cells = body.launch.description.launchOrder.filter(
    (process): process is PythonRemoteStageLaunch =>
      process.kind === "remote-stage" && process.cell !== null,
  );
  const cell = cells.length === 1 ? cells[0]!.cell : null;
  const cellLaunch = cells.length === 1 ? cells[0]! : null;

  const physicalMeasurement =
    body.provenance.level === "hardware-physical" &&
    body.provenance.source === "measurement" &&
    body.provenance.networkScope !== "loopback" &&
    !body.provenance.loopback &&
    !body.provenance.emulated;
  const sealedLaunch = body.launch.canonicalSha256 === launchHash;
  const twoDistinctHosts =
    hosts.length === 2 &&
    uniqueCount(hostIds) === 2 &&
    uniqueCount(hostFingerprints) === 2 &&
    uniqueCount(hosts.map((host) => host.agentId)) === 2 &&
    uniqueCount(hosts.map((host) => normalizedEndpointOrigin(host.agentEndpoint))) === 2;
  const twoDistinctGpus =
    hosts.length === 2 &&
    uniqueCount(gpuFingerprints) === 2 &&
    hosts.every(
      (host) =>
        host.gpu.physicalVramBytes > 0 &&
        host.gpu.offeredVramBytes > 0 &&
        host.gpu.offeredVramBytes <= host.gpu.physicalVramBytes,
    );

  const routeHosts = [
    ...hosts.map((host) => endpointHostname(host.agentEndpoint)),
    ...(cell?.external
      ? [cell.external.controlAdvertiseHost, cell.external.distributedAdvertiseHost]
      : []),
    ...(cellLaunch?.members.map((member) => member.endpoint.host) ?? []),
  ];
  const nonLoopbackRoute =
    routeHosts.length >= 4 && routeHosts.every((host) => isRemotelyRoutableHost(host));

  const gpuCollective =
    cells.length === 1 &&
    cell !== null &&
    cell.fixture.location === "member-local" &&
    cell.external !== undefined &&
    cell.worldSize === 2 &&
    cell.collectiveBackend === "nccl" &&
    cell.rankMemberIds.length === 2 &&
    cell.rankDevices.length === 2 &&
    cell.rankDevices.every((device) => /^cuda:[0-9]+$/.test(device)) &&
    hosts.length === 2 &&
    hosts.every(
      (host) =>
        (host.gpu.computeApi === "cuda" || host.gpu.computeApi === "rocm") &&
        host.gpu.runtimeAvailable &&
        host.gpu.collectiveAvailable,
    );

  const hostByNode = new Map(hosts.map((host) => [host.rankNodeId, host]));
  const oneRankPerHost =
    cell !== null &&
    cell.rankMemberIds.length === 2 &&
    cell.rankMemberIds.every((nodeId) => hostByNode.has(nodeId)) &&
    uniqueCount(cell.rankMemberIds.map((nodeId) => hostByNode.get(nodeId)?.hostId ?? "")) === 2;

  const workByRank = new Map(body.rankWork.map((work) => [work.rank, work]));
  const bothRanksExecuted =
    cell !== null &&
    body.rankWork.length === 2 &&
    workByRank.size === 2 &&
    [0, 1].every((rank) => {
      const work = workByRank.get(rank);
      const nodeId = cell.rankMemberIds[rank];
      const device = cell.rankDevices[rank];
      const host = nodeId === undefined ? undefined : hostByNode.get(nodeId);
      return (
        work !== undefined &&
        host !== undefined &&
        work.nodeId === nodeId &&
        work.hostId === host.hostId &&
        work.device === device &&
        host.gpu.device === device &&
        work.forwardCalls > 0 &&
        work.collectiveCalls > 0 &&
        work.tokensProcessed > 0 &&
        work.peakAllocatedBytes > 0 &&
        work.peakAllocatedBytes <= host.gpu.offeredVramBytes
      );
    });

  const directionalNetworkMeasured = hasMeasuredDirectionalNetwork(
    body.networkLinks,
    hosts,
    body.policy.minimumDirectionalLinkSamples,
  );
  const exactTokenParity = hasExactTokenParity(body);
  const sufficientSamples = hasSufficientSamples(body.samples, body.policy);
  const healthyCleanLifecycle = hasHealthyCleanLifecycle(body);

  const checks: PhysicalGateCheckV1[] = [
    result("physical_measurement", physicalMeasurement, "requires_real_non_emulated_measurement"),
    result("sealed_launch", sealedLaunch, "launch_description_hash_mismatch"),
    result("two_distinct_hosts", twoDistinctHosts, "requires_two_distinct_host_identities"),
    result("two_distinct_gpus", twoDistinctGpus, "requires_two_distinct_physical_gpus"),
    result("non_loopback_route", nonLoopbackRoute, "route_contains_local_or_unspecified_host"),
    result("two_rank_gpu_collective", gpuCollective, "requires_external_two_rank_nccl_or_rccl_cell"),
    result("one_rank_per_host", oneRankPerHost, "collective_ranks_are_not_mapped_one_per_host"),
    result("both_ranks_executed", bothRanksExecuted, "one_or_more_ranks_have_no_verified_work"),
    result(
      "directional_network_measured",
      directionalNetworkMeasured,
      "missing_bidirectional_raw_network_samples",
    ),
    result("exact_token_parity", exactTokenParity, "distributed_token_ids_differ_from_reference"),
    result("sufficient_samples", sufficientSamples, "benchmark_sample_floor_not_met"),
    result("healthy_clean_lifecycle", healthyCleanLifecycle, "runtime_not_ready_or_cleanup_incomplete"),
  ];
  return { passed: checks.every((check) => check.passed), checks };
}

function result(id: PhysicalGateCheckId, passed: boolean, failure: string): PhysicalGateCheckV1 {
  return { id, passed, detail: passed ? "ok" : failure };
}

function hasMeasuredDirectionalNetwork(
  links: PhysicalGateDirectionalLinkEvidenceV1[],
  hosts: PhysicalGateHostEvidenceV1[],
  minimumSamples: number,
): boolean {
  if (hosts.length !== 2 || links.length !== 2) return false;
  const [left, right] = hosts;
  if (left === undefined || right === undefined) return false;
  const required = new Set([`${left.hostId}\0${right.hostId}`, `${right.hostId}\0${left.hostId}`]);
  const observed = new Set<string>();
  for (const link of links) {
    const key = `${link.fromHostId}\0${link.toHostId}`;
    if (!required.has(key) || observed.has(key)) return false;
    if (
      link.rttMs.length < minimumSamples ||
      link.goodputMbps.length < minimumSamples ||
      link.rttMs.some((value) => value <= 0) ||
      link.goodputMbps.some((value) => value <= 0)
    ) {
      return false;
    }
    observed.add(key);
  }
  return observed.size === required.size;
}

function hasExactTokenParity(body: Omit<PhysicalGateReportBodyV1, "gate">): boolean {
  const reference = body.reference;
  if (
    reference.mode !== "monolithic-greedy" ||
    reference.modelId !== body.launch.description.modelIdentity.id ||
    reference.modelRevision !== body.launch.description.modelIdentity.revision ||
    reference.tokenizerId !== body.launch.description.modelIdentity.tokenizerId ||
    reference.outputTokenIds.length === 0 ||
    reference.outputTokenIdsHashScheme !== OUTPUT_TOKEN_IDS_HASH_SCHEME ||
    reference.outputTokenIdsSha256 !== outputTokenIdsSha256(reference.outputTokenIds)
  ) {
    return false;
  }
  const measured = body.samples.filter((sample) => sample.phase === "measure");
  return (
    measured.length > 0 &&
    measured.every(
      (sample) =>
        sample.outputTokenIdsHashScheme === OUTPUT_TOKEN_IDS_HASH_SCHEME &&
        sample.completionTokens === reference.outputTokenIds.length &&
        sample.outputTokenIdsSha256 === reference.outputTokenIdsSha256,
    )
  );
}

function hasSufficientSamples(
  samples: PhysicalGateRequestSampleV1[],
  policy: PhysicalTwoHostGpuGatePolicyV1,
): boolean {
  const warmups = samples.filter((sample) => sample.phase === "warmup");
  const measured = samples.filter((sample) => sample.phase === "measure");
  return (
    uniqueCount(samples.map((sample) => sample.sampleId)) === samples.length &&
    warmups.length >= policy.minimumWarmupSamples &&
    measured.length >= policy.minimumMeasuredSamples &&
    measured.every(
      (sample) =>
        sample.completionTokens >= policy.minimumCompletionTokensPerMeasuredSample &&
        sample.responseMs > 0 &&
        sample.pipelineMs > 0 &&
        sample.ttftMs >= 0 &&
        sample.ttftMs <= sample.responseMs &&
        sample.tpotMs >= 0,
    )
  );
}

function hasHealthyCleanLifecycle(body: Omit<PhysicalGateReportBodyV1, "gate">): boolean {
  const expectedProcesses = body.launch.description.launchOrder.map((process) => process.processId);
  const expectedBoundaries = body.launch.description.launchOrder.find(
    (process) => process.kind === "root-engine",
  )?.boundaries;
  const lifecycle = body.lifecycle;
  return (
    sameStringSet(lifecycle.readyProcessIds, expectedProcesses) &&
    sameStringSet(lifecycle.stoppedProcessIds, expectedProcesses) &&
    lifecycle.residualProcessIds.length === 0 &&
    lifecycle.health.status === "ready" &&
    lifecycle.health.model === body.launch.description.configuration.publicModelName &&
    lifecycle.health.stages === body.launch.description.route.logicalStageIds.length &&
    canonicalEvidenceJson(lifecycle.health.boundaries) ===
      canonicalEvidenceJson(expectedBoundaries ?? []) &&
    lifecycle.health.codec === body.launch.description.route.codec
  );
}

function summarizeSamples(samples: PhysicalGateRequestSampleV1[]): PhysicalGateMetricSummaryV1 {
  const measured = samples.filter((sample) => sample.phase === "measure");
  return {
    warmupSamples: samples.length - measured.length,
    measuredSamples: measured.length,
    measuredCompletionTokens: measured.reduce(
      (total, sample) => total + sample.completionTokens,
      0,
    ),
    ttftMs: describe(measured.map((sample) => sample.ttftMs)),
    tpotMs: describe(measured.map((sample) => sample.tpotMs)),
    responseMs: describe(measured.map((sample) => sample.responseMs)),
    pipelineMs: describe(measured.map((sample) => sample.pipelineMs)),
    outputTokensPerSecondIncludingTtft: describe(
      measured.map((sample) => sample.completionTokens / (sample.responseMs / 1_000)),
    ),
  };
}

function describe(values: number[]): PhysicalGateMetricStatsV1 {
  if (values.length === 0) {
    return { count: 0, mean: null, p50: null, p95: null, min: null, max: null };
  }
  const sorted = [...values].sort((left, right) => left - right);
  return {
    count: sorted.length,
    mean: sorted.reduce((total, value) => total + value, 0) / sorted.length,
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    min: sorted[0]!,
    max: sorted.at(-1)!,
  };
}

function percentile(sorted: number[], fraction: number): number {
  if (sorted.length === 1) return sorted[0]!;
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower]!;
  const weight = position - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}

function assertReport(value: unknown): asserts value is PhysicalTwoHostGpuGateReportV1 {
  const report = record(value, "physical_gpu_gate_report");
  exactKeys(
    report,
    [
      "schema",
      "capturedAt",
      "provenance",
      "policy",
      "launch",
      "hosts",
      "networkLinks",
      "lifecycle",
      "reference",
      "rankWork",
      "samples",
      "summary",
      "gate",
      "seal",
    ],
    "physical_gpu_gate_report",
  );
  if (report.schema !== PHYSICAL_TWO_HOST_GPU_GATE_SCHEMA) {
    throw new Error("physical_gpu_gate_unsupported_schema");
  }
  assertEvidence(report);
  assertPolicy(report.policy);
  assertSummary(report.summary);
  assertGate(report.gate);
  assertSeal(report.seal);
}

function assertEvidence(
  value: unknown,
  requireExactInputKeys = false,
): asserts value is PhysicalTwoHostGpuGateEvidenceV1 {
  const evidence = record(value, "physical_gpu_gate_evidence");
  if (requireExactInputKeys) {
    exactKeys(
      evidence,
      [
        "capturedAt",
        "provenance",
        "launch",
        "hosts",
        "networkLinks",
        "lifecycle",
        "reference",
        "rankWork",
        "samples",
      ],
      "physical_gpu_gate_evidence",
    );
  }
  isoTimestamp(evidence.capturedAt, "physical_gpu_gate_captured_at");
  assertProvenance(evidence.provenance);
  assertLaunch(evidence.launch);
  array(evidence.hosts, 0, MAX_HOSTS, "physical_gpu_gate_hosts").forEach(assertHost);
  array(evidence.networkLinks, 0, MAX_LINKS, "physical_gpu_gate_network_links").forEach(
    assertNetworkLink,
  );
  assertLifecycle(evidence.lifecycle);
  assertReference(evidence.reference);
  array(evidence.rankWork, 0, MAX_RANKS, "physical_gpu_gate_rank_work").forEach(assertRankWork);
  array(evidence.samples, 0, MAX_SAMPLES, "physical_gpu_gate_samples").forEach(assertSample);
}

function assertProvenance(value: unknown): void {
  const item = record(value, "physical_gpu_gate_provenance");
  exactKeys(
    item,
    ["level", "source", "networkScope", "loopback", "emulated", "attestation"],
    "physical_gpu_gate_provenance",
  );
  literal(item.level, "hardware-physical", "physical_gpu_gate_evidence_level");
  oneOf(item.source, ["measurement", "projection"], "physical_gpu_gate_evidence_source");
  oneOf(item.networkScope, ["lan", "wan", "loopback"], "physical_gpu_gate_network_scope");
  boolean(item.loopback, "physical_gpu_gate_loopback");
  boolean(item.emulated, "physical_gpu_gate_emulated");
  literal(item.attestation, "self-reported", "physical_gpu_gate_attestation");
}

function assertPolicy(value: unknown): asserts value is PhysicalTwoHostGpuGatePolicyV1 {
  const policy = record(value, "physical_gpu_gate_policy");
  exactKeys(
    policy,
    [
      "schema",
      "minimumWarmupSamples",
      "minimumMeasuredSamples",
      "minimumCompletionTokensPerMeasuredSample",
      "minimumDirectionalLinkSamples",
    ],
    "physical_gpu_gate_policy",
  );
  if (canonicalEvidenceJson(policy) !== canonicalEvidenceJson(PHYSICAL_TWO_HOST_GPU_GATE_POLICY)) {
    throw new Error("physical_gpu_gate_policy_mismatch");
  }
}

function assertLaunch(value: unknown): void {
  const launch = record(value, "physical_gpu_gate_launch");
  exactKeys(launch, ["description", "canonicalSha256"], "physical_gpu_gate_launch");
  record(launch.description, "physical_gpu_gate_launch_description");
  sha256(launch.canonicalSha256, "physical_gpu_gate_launch_sha256");
}

function assertHost(value: unknown): void {
  const host = record(value, "physical_gpu_gate_host");
  exactKeys(
    host,
    ["hostId", "hostFingerprintSha256", "agentId", "agentEndpoint", "rankNodeId", "gpu"],
    "physical_gpu_gate_host",
  );
  text(host.hostId, "physical_gpu_gate_host_id");
  sha256(host.hostFingerprintSha256, "physical_gpu_gate_host_fingerprint");
  text(host.agentId, "physical_gpu_gate_agent_id");
  httpEndpoint(host.agentEndpoint, "physical_gpu_gate_agent_endpoint");
  text(host.rankNodeId, "physical_gpu_gate_rank_node_id");
  assertGpu(host.gpu);
}

function assertGpu(value: unknown): void {
  const gpu = record(value, "physical_gpu_gate_gpu");
  exactKeys(
    gpu,
    [
      "deviceFingerprintSha256",
      "device",
      "vendor",
      "model",
      "physicalVramBytes",
      "offeredVramBytes",
      "computeApi",
      "runtimeAvailable",
      "collectiveAvailable",
    ],
    "physical_gpu_gate_gpu",
  );
  sha256(gpu.deviceFingerprintSha256, "physical_gpu_gate_gpu_fingerprint");
  text(gpu.device, "physical_gpu_gate_gpu_device");
  text(gpu.vendor, "physical_gpu_gate_gpu_vendor");
  text(gpu.model, "physical_gpu_gate_gpu_model");
  integer(gpu.physicalVramBytes, 0, Number.MAX_SAFE_INTEGER, "physical_gpu_gate_gpu_vram");
  integer(gpu.offeredVramBytes, 0, Number.MAX_SAFE_INTEGER, "physical_gpu_gate_gpu_offer");
  oneOf(gpu.computeApi, ["cuda", "rocm", "cpu"], "physical_gpu_gate_gpu_compute_api");
  boolean(gpu.runtimeAvailable, "physical_gpu_gate_gpu_runtime_available");
  boolean(gpu.collectiveAvailable, "physical_gpu_gate_gpu_collective_available");
}

function assertNetworkLink(value: unknown): void {
  const link = record(value, "physical_gpu_gate_network_link");
  exactKeys(
    link,
    ["fromHostId", "toHostId", "rttMs", "goodputMbps"],
    "physical_gpu_gate_network_link",
  );
  text(link.fromHostId, "physical_gpu_gate_link_from");
  text(link.toHostId, "physical_gpu_gate_link_to");
  numberArray(link.rttMs, "physical_gpu_gate_link_rtt");
  numberArray(link.goodputMbps, "physical_gpu_gate_link_goodput");
}

function assertLifecycle(value: unknown): void {
  const lifecycle = record(value, "physical_gpu_gate_lifecycle");
  exactKeys(
    lifecycle,
    ["readyProcessIds", "stoppedProcessIds", "residualProcessIds", "health"],
    "physical_gpu_gate_lifecycle",
  );
  stringArray(lifecycle.readyProcessIds, "physical_gpu_gate_ready_processes");
  stringArray(lifecycle.stoppedProcessIds, "physical_gpu_gate_stopped_processes");
  stringArray(lifecycle.residualProcessIds, "physical_gpu_gate_residual_processes");
  const health = record(lifecycle.health, "physical_gpu_gate_health");
  exactKeys(health, ["status", "model", "stages", "boundaries", "codec"], "physical_gpu_gate_health");
  text(health.status, "physical_gpu_gate_health_status");
  text(health.model, "physical_gpu_gate_health_model");
  integer(health.stages, 1, 1_024, "physical_gpu_gate_health_stages");
  integerArray(health.boundaries, "physical_gpu_gate_health_boundaries");
  text(health.codec, "physical_gpu_gate_health_codec");
}

function assertReference(value: unknown): void {
  const reference = record(value, "physical_gpu_gate_reference");
  exactKeys(
    reference,
    [
      "mode",
      "modelId",
      "modelRevision",
      "tokenizerId",
      "promptTokenIdsSha256",
      "outputTokenIds",
      "outputTokenIdsHashScheme",
      "outputTokenIdsSha256",
    ],
    "physical_gpu_gate_reference",
  );
  literal(reference.mode, "monolithic-greedy", "physical_gpu_gate_reference_mode");
  text(reference.modelId, "physical_gpu_gate_reference_model");
  text(reference.modelRevision, "physical_gpu_gate_reference_revision");
  text(reference.tokenizerId, "physical_gpu_gate_reference_tokenizer");
  sha256(reference.promptTokenIdsSha256, "physical_gpu_gate_prompt_sha256");
  tokenIds(reference.outputTokenIds, "physical_gpu_gate_reference_tokens");
  literal(
    reference.outputTokenIdsHashScheme,
    OUTPUT_TOKEN_IDS_HASH_SCHEME,
    "physical_gpu_gate_reference_token_hash_scheme",
  );
  sha256(reference.outputTokenIdsSha256, "physical_gpu_gate_reference_tokens_sha256");
}

function assertRankWork(value: unknown): void {
  const work = record(value, "physical_gpu_gate_rank_work_item");
  exactKeys(
    work,
    [
      "rank",
      "hostId",
      "nodeId",
      "device",
      "forwardCalls",
      "collectiveCalls",
      "tokensProcessed",
      "bytesSent",
      "bytesReceived",
      "peakAllocatedBytes",
    ],
    "physical_gpu_gate_rank_work_item",
  );
  integer(work.rank, 0, 255, "physical_gpu_gate_rank");
  text(work.hostId, "physical_gpu_gate_rank_host");
  text(work.nodeId, "physical_gpu_gate_rank_node");
  text(work.device, "physical_gpu_gate_rank_device");
  for (const key of [
    "forwardCalls",
    "collectiveCalls",
    "tokensProcessed",
    "bytesSent",
    "bytesReceived",
    "peakAllocatedBytes",
  ] as const) {
    integer(work[key], 0, Number.MAX_SAFE_INTEGER, `physical_gpu_gate_rank_${key}`);
  }
}

function assertSample(value: unknown): void {
  const sample = record(value, "physical_gpu_gate_sample");
  exactKeys(
    sample,
    [
      "sampleId",
      "phase",
      "concurrency",
      "iteration",
      "promptTokens",
      "completionTokens",
      "outputTokenIdsHashScheme",
      "outputTokenIdsSha256",
      "ttftMs",
      "tpotMs",
      "responseMs",
      "pipelineMs",
    ],
    "physical_gpu_gate_sample",
  );
  text(sample.sampleId, "physical_gpu_gate_sample_id");
  oneOf(sample.phase, ["warmup", "measure"], "physical_gpu_gate_sample_phase");
  integer(sample.concurrency, 1, 65_536, "physical_gpu_gate_sample_concurrency");
  integer(sample.iteration, 0, Number.MAX_SAFE_INTEGER, "physical_gpu_gate_sample_iteration");
  integer(sample.promptTokens, 1, MAX_TOKEN_IDS, "physical_gpu_gate_sample_prompt_tokens");
  integer(sample.completionTokens, 0, MAX_TOKEN_IDS, "physical_gpu_gate_sample_completion_tokens");
  literal(
    sample.outputTokenIdsHashScheme,
    OUTPUT_TOKEN_IDS_HASH_SCHEME,
    "physical_gpu_gate_sample_output_hash_scheme",
  );
  sha256(sample.outputTokenIdsSha256, "physical_gpu_gate_sample_output_sha256");
  nonNegative(sample.ttftMs, "physical_gpu_gate_sample_ttft");
  nonNegative(sample.tpotMs, "physical_gpu_gate_sample_tpot");
  nonNegative(sample.responseMs, "physical_gpu_gate_sample_response");
  nonNegative(sample.pipelineMs, "physical_gpu_gate_sample_pipeline");
}

function assertSummary(value: unknown): void {
  const summary = record(value, "physical_gpu_gate_summary");
  exactKeys(
    summary,
    [
      "warmupSamples",
      "measuredSamples",
      "measuredCompletionTokens",
      "ttftMs",
      "tpotMs",
      "responseMs",
      "pipelineMs",
      "outputTokensPerSecondIncludingTtft",
    ],
    "physical_gpu_gate_summary",
  );
  integer(summary.warmupSamples, 0, MAX_SAMPLES, "physical_gpu_gate_summary_warmups");
  integer(summary.measuredSamples, 0, MAX_SAMPLES, "physical_gpu_gate_summary_measured");
  integer(
    summary.measuredCompletionTokens,
    0,
    Number.MAX_SAFE_INTEGER,
    "physical_gpu_gate_summary_tokens",
  );
  for (const key of [
    "ttftMs",
    "tpotMs",
    "responseMs",
    "pipelineMs",
    "outputTokensPerSecondIncludingTtft",
  ] as const) {
    assertStats(summary[key], `physical_gpu_gate_summary_${key}`);
  }
}

function assertStats(value: unknown, name: string): void {
  const stats = record(value, name);
  exactKeys(stats, ["count", "mean", "p50", "p95", "min", "max"], name);
  integer(stats.count, 0, MAX_SAMPLES, `${name}_count`);
  for (const key of ["mean", "p50", "p95", "min", "max"] as const) {
    nullableNonNegative(stats[key], `${name}_${key}`);
  }
}

function assertGate(value: unknown): void {
  const gate = record(value, "physical_gpu_gate_decision");
  exactKeys(gate, ["passed", "checks"], "physical_gpu_gate_decision");
  boolean(gate.passed, "physical_gpu_gate_passed");
  const checks = array(gate.checks, 0, 64, "physical_gpu_gate_checks");
  for (const value of checks) {
    const check = record(value, "physical_gpu_gate_check");
    exactKeys(check, ["id", "passed", "detail"], "physical_gpu_gate_check");
    oneOf(
      check.id,
      [
        "physical_measurement",
        "sealed_launch",
        "two_distinct_hosts",
        "two_distinct_gpus",
        "non_loopback_route",
        "two_rank_gpu_collective",
        "one_rank_per_host",
        "both_ranks_executed",
        "directional_network_measured",
        "exact_token_parity",
        "sufficient_samples",
        "healthy_clean_lifecycle",
      ],
      "physical_gpu_gate_check_id",
    );
    boolean(check.passed, "physical_gpu_gate_check_passed");
    text(check.detail, "physical_gpu_gate_check_detail");
  }
}

function assertSeal(value: unknown): void {
  const seal = record(value, "physical_gpu_gate_seal");
  exactKeys(seal, ["canonicalization", "algorithm", "digest"], "physical_gpu_gate_seal");
  literal(
    seal.canonicalization,
    PHYSICAL_GPU_GATE_CANONICALIZATION,
    "physical_gpu_gate_canonicalization",
  );
  literal(seal.algorithm, "sha256", "physical_gpu_gate_seal_algorithm");
  sha256(seal.digest, "physical_gpu_gate_seal_digest");
}

function validateLaunch(value: unknown): asserts value is PythonPipelineLaunchDescription {
  try {
    validatePythonLaunchDescription(value);
  } catch (error) {
    throw new Error("physical_gpu_gate_launch_description_is_invalid", { cause: error });
  }
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${name}_must_be_an_object`);
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, expected: string[], name: string): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (canonicalEvidenceJson(actual) !== canonicalEvidenceJson(wanted)) {
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
  if (typeof value !== "string" || value.length < 1 || value.length > 4_096) {
    throw new Error(`${name}_must_be_text`);
  }
}

function literal<T extends string>(value: unknown, expected: T, name: string): asserts value is T {
  if (value !== expected) throw new Error(`${name}_is_invalid`);
}

function oneOf<T extends string>(
  value: unknown,
  expected: readonly T[],
  name: string,
): asserts value is T {
  if (typeof value !== "string" || !expected.includes(value as T)) {
    throw new Error(`${name}_is_invalid`);
  }
}

function boolean(value: unknown, name: string): asserts value is boolean {
  if (typeof value !== "boolean") throw new Error(`${name}_must_be_boolean`);
}

function integer(value: unknown, minimum: number, maximum: number, name: string): asserts value is number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new Error(`${name}_must_be_an_integer`);
  }
}

function nonNegative(value: unknown, name: string): asserts value is number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(`${name}_must_be_non_negative`);
  }
}

function nullableNonNegative(value: unknown, name: string): void {
  if (value === null) return;
  nonNegative(value, name);
}

function sha256(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !SHA256_PATTERN.test(value)) {
    throw new Error(`${name}_must_be_sha256`);
  }
}

function isoTimestamp(value: unknown, name: string): asserts value is string {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value)) {
    throw new Error(`${name}_must_be_utc_iso8601`);
  }
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.valueOf()) || parsed.toISOString() !== value) {
    throw new Error(`${name}_must_be_utc_iso8601`);
  }
}

function httpEndpoint(value: unknown, name: string): asserts value is string {
  text(value, name);
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${name}_must_be_http_url`);
  }
  if (
    (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.search !== "" ||
    parsed.hash !== "" ||
    (parsed.pathname !== "/" && parsed.pathname !== "")
  ) {
    throw new Error(`${name}_must_be_http_url`);
  }
}

function endpointHostname(endpoint: string): string {
  try {
    return new URL(endpoint).hostname;
  } catch {
    return "";
  }
}

function normalizedEndpointOrigin(endpoint: string): string {
  try {
    return new URL(endpoint).origin.toLowerCase();
  } catch {
    return "";
  }
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
  if (ipv4) {
    const octets = ipv4.slice(1).map(Number);
    if (octets.some((part) => part > 255)) return false;
    if (octets[0] === 127 || octets.every((part) => part === 0)) return false;
  }
  return true;
}

function stringArray(value: unknown, name: string): string[] {
  const result = array(value, 0, 100_000, name);
  for (const item of result) text(item, name);
  return result as string[];
}

function numberArray(value: unknown, name: string): number[] {
  const result = array(value, 0, 100_000, name);
  for (const item of result) nonNegative(item, name);
  return result as number[];
}

function integerArray(value: unknown, name: string): number[] {
  const result = array(value, 0, 4_096, name);
  for (const item of result) integer(item, 0, Number.MAX_SAFE_INTEGER, name);
  return result as number[];
}

function tokenIds(value: unknown, name: string): number[] {
  const result = array(value, 0, MAX_TOKEN_IDS, name);
  for (const item of result) integer(item, 0, 0xffff_ffff, name);
  return result as number[];
}

function uniqueCount(values: string[]): number {
  return new Set(values).size;
}

function sameStringSet(left: string[], right: string[]): boolean {
  return (
    left.length === right.length &&
    new Set(left).size === left.length &&
    new Set(right).size === right.length &&
    [...left].sort().every((value, index) => value === [...right].sort()[index])
  );
}
