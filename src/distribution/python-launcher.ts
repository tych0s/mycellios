import { createHash } from "node:crypto";
import {
  validateRuntimePipelineManifest,
  type RuntimeActivationCodec,
  type RuntimeDecodePlanManifest,
  type RuntimeEndpoint,
  type RuntimePipelineManifestV2,
  type RuntimePrefillPlanManifest,
  type RuntimeSpeculationPolicy,
  type RuntimeStageMemberManifest,
  type RuntimeTensorParallelCellExecutionManifest,
  type RuntimeVirtualStageManifest,
} from "./runtime-manifest.js";

export type PythonLaunchPhase = "prefill" | "decode";

export interface PythonRuntimeModelInput {
  /** Hugging Face id or local snapshot path consumed by the Python loaders. */
  source: string;
  /** Null deliberately omits --revision, as required by local snapshots. */
  revision?: string | null;
  /** Decimal uint64 returned by Python model_snapshot_identity(). */
  snapshotIdentity?: string;
}

export interface PythonLaunchCompilerOptions {
  /** Bind address for the OpenAI-compatible root server. */
  apiEndpoint: RuntimeEndpoint;
  /** Address advertised to remote stages for ACK/token return. */
  returnEndpoint: RuntimeEndpoint;
  /** Local interface used by the root return listener, e.g. 0.0.0.0. */
  returnBindHost: string;
  runtimeModel?: PythonRuntimeModelInput;
  publicModelName?: string;
  pythonExecutable?: string;
  stageModule?: string;
  serverModule?: string;
  threadsPerStage?: number;
  connectTimeoutSeconds?: number;
  batchWindowMs?: number;
  maxPendingRequests?: number;
  maxOutputTokens?: number;
  speculationMinimumSpeedup?: number;
}

export interface PythonRuntimeModelSource {
  source: string;
  revision: string | null;
  snapshotIdentity?: string;
}

export interface PythonLaunchConfiguration {
  apiEndpoint: RuntimeEndpoint;
  returnEndpoint: RuntimeEndpoint;
  returnBindHost: string;
  runtimeModel: PythonRuntimeModelSource;
  publicModelName: string;
  pythonExecutable: string;
  stageModule: string;
  serverModule: string;
  threadsPerStage: number;
  connectTimeoutSeconds: number;
  batchWindowMs: number;
  maxPendingRequests: number;
  maxOutputTokens: number;
  speculationMinimumSpeedup: number;
}

export interface PythonPrefillLaunchSettings {
  phase: "prefill";
  planId: string;
  activationCodec: RuntimeActivationCodec;
  microBatchSize: number;
  chunkTokens: number;
}

export interface PythonDecodeLaunchSettings {
  phase: "decode";
  planId: string;
  activationCodec: RuntimeActivationCodec;
  microBatchSize: number;
  directTokenReturnStage: number;
  speculation: RuntimeSpeculationPolicy;
}

export type PythonPhaseLaunchSettings =
  | PythonPrefillLaunchSettings
  | PythonDecodeLaunchSettings;

export interface PythonStageCommand {
  executable: string;
  /** Argument vector intended for execFile/spawn with shell=false. */
  args: string[];
}

export interface PythonDownstreamStage {
  stageId: string;
  stageIndex: number;
  layerEnd: number;
  anchorMemberId: string;
  endpoint: RuntimeEndpoint;
}

interface PythonLaunchBase {
  launchIndex: number;
  processId: string;
  routeId: string;
  phases: ["prefill", "decode"];
  logicalPlanIds: [string, string];
  stageId: string;
  stageIndex: number;
  layerStart: number;
  layerEnd: number;
  totalLayers: number;
  codec: RuntimeActivationCodec;
  anchor: {
    memberId: string;
    endpoint: RuntimeEndpoint;
  };
  members: RuntimeStageMemberManifest[];
  command: PythonStageCommand;
}

/** A physical stage_cli process. Stage zero can never use this variant. */
export interface PythonRemoteStageLaunch extends PythonLaunchBase {
  kind: "remote-stage";
  downstream: PythonDownstreamStage | null;
  returnEndpoint: RuntimeEndpoint;
  /** Null selects the ordinary single-member StageModelRunner. */
  cell: RuntimeTensorParallelCellExecutionManifest | null;
}

/** One nonzero rank that joins a member-local TP cell before its anchor is ready. */
export interface PythonCellMemberLaunch extends PythonLaunchBase {
  kind: "cell-member";
  rank: number;
  fixturePath: string;
  pipelineSnapshotIdentity: string;
  collectiveBackend: RuntimeTensorParallelCellExecutionManifest["collectiveBackend"];
  computeDtype: RuntimeTensorParallelCellExecutionManifest["computeDtype"];
  device: string;
  cellAnchor: {
    memberId: string;
    controlEndpoint: RuntimeEndpoint;
  };
  worldSize: number;
  operationTimeoutSeconds: number;
  startupTimeoutSeconds: number;
}

/** Root partition hosted by DistributedPipelineEngine inside server.py. */
export interface PythonRootEngineLaunch extends PythonLaunchBase {
  kind: "root-engine";
  stageIndex: 0;
  boundaries: number[];
  firstRemoteStage: PythonDownstreamStage;
  apiEndpoint: RuntimeEndpoint;
  returnBindHost: string;
  returnEndpoint: RuntimeEndpoint;
  prefill: PythonPrefillLaunchSettings;
  decode: PythonDecodeLaunchSettings;
}

export type PythonLaunchProcess =
  | PythonCellMemberLaunch
  | PythonRemoteStageLaunch
  | PythonRootEngineLaunch;

export interface PythonPipelineLaunchRoute {
  routeId: string;
  phases: ["prefill", "decode"];
  phaseSettings: [PythonPrefillLaunchSettings, PythonDecodeLaunchSettings];
  codec: RuntimeActivationCodec;
  /** Logical model order, root first. */
  logicalStageIds: string[];
  /** Remote stages downstream-first, then the root server last. */
  downstreamFirstProcessIds: string[];
  cellMemberProcessIds: string[];
  remoteStageProcessIds: string[];
  rootProcessId: string;
}

export interface PythonPipelineLaunchDescription {
  schema: "gdlp-python-launch/2";
  launchId: string;
  sourceProtocol: "gdlp/2";
  sourceManifest: RuntimePipelineManifestV2;
  pipelineId: string;
  modelIdentity: {
    id: string;
    revision: string;
    tokenizerId: string;
  };
  runtimeModel: PythonRuntimeModelSource;
  totalLayers: number;
  configuration: PythonLaunchConfiguration;
  route: PythonPipelineLaunchRoute;
  launchOrder: PythonLaunchProcess[];
}

interface PythonSpeculationArguments {
  provider: "off" | "ngram";
  maxDraftTokens: number;
}

/**
 * Produce a launch plan only. This module never imports child_process and
 * never starts a Python process.
 */
export function compilePythonLaunchDescription(
  manifestValue: unknown,
  optionsValue: PythonLaunchCompilerOptions,
): PythonPipelineLaunchDescription {
  validateRuntimePipelineManifest(manifestValue);
  if (manifestValue.protocol !== "gdlp/2") {
    throw new Error("python_launcher_requires_gdlp_2");
  }
  const configuration = normalizeConfiguration(manifestValue, optionsValue);
  const description = buildDescription(manifestValue, configuration);
  validatePythonLaunchDescription(description);
  return description;
}

/**
 * Closed validation regenerates every derived id, edge and argv from the
 * embedded validated manifest and normalized configuration.
 */
export function validatePythonLaunchDescription(
  value: unknown,
): asserts value is PythonPipelineLaunchDescription {
  if (!isRecord(value)) throw new Error("python_launch_description_must_be_an_object");
  if (value.schema !== "gdlp-python-launch/2") {
    throw new Error("unsupported_python_launch_schema");
  }
  validateRuntimePipelineManifest(value.sourceManifest);
  if (value.sourceManifest.protocol !== "gdlp/2") {
    throw new Error("python_launcher_requires_gdlp_2");
  }
  const configuration = normalizeConfiguration(
    value.sourceManifest,
    value.configuration as unknown as PythonLaunchCompilerOptions,
    true,
  );
  const expected = buildDescription(value.sourceManifest, configuration);
  if (canonicalJson(value) !== canonicalJson(expected)) {
    throw new Error("python_launch_description_mismatch");
  }
}

function buildDescription(
  manifest: RuntimePipelineManifestV2,
  configuration: PythonLaunchConfiguration,
): PythonPipelineLaunchDescription {
  const prefill = manifest.plans.prefill;
  const decode = manifest.plans.decode;
  assertSharedExecutableRoute(manifest, prefill, decode);
  const stages = prefill.stages;
  const phaseSettings: [PythonPrefillLaunchSettings, PythonDecodeLaunchSettings] = [
    prefillSettings(prefill),
    decodeSettings(decode),
  ];
  const codec = prefill.activationCodec;
  const routeId = routeIdentity(manifest.pipelineId, phaseSettings, stages, codec);
  const planIds: [string, string] = [prefill.planId, decode.planId];
  const boundaries = [0, ...stages.map((stage) => stage.layerEnd)];
  const launchOrder: PythonLaunchProcess[] = [];
  const occupiedRemoteEndpoints = new Set<string>();

  // Stage N-1 must listen before N-2 connects, continuing back to stage 1.
  for (let index = stages.length - 1; index >= 1; index -= 1) {
    const stage = stages[index]!;
    const listenKey = endpointKey(stage.anchor.endpoint);
    if (occupiedRemoteEndpoints.has(listenKey)) {
      throw new Error("python_remote_stage_endpoint_reused");
    }
    if (
      listenKey === endpointKey(configuration.apiEndpoint) ||
      listenKey === endpointKey(configuration.returnEndpoint)
    ) {
      throw new Error("python_remote_stage_endpoint_conflicts_with_root");
    }
    occupiedRemoteEndpoints.add(listenKey);
    const next = stages[index + 1];
    const downstream = next ? downstreamStage(next) : null;
    if (stage.execution?.fixture.location === "member-local") {
      const external = stage.execution.external!;
      const pipelineSnapshotIdentity = configuration.runtimeModel.snapshotIdentity;
      if (!pipelineSnapshotIdentity) {
        throw new Error("python_external_cell_requires_snapshot_identity");
      }
      for (let rank = 1; rank < stage.execution.worldSize; rank += 1) {
        const memberId = stage.execution.rankMemberIds[rank]!;
        const member = stage.members.find((candidate) => candidate.nodeId === memberId);
        if (!member) throw new Error("python_external_cell_rank_member_is_missing");
        const memberPartial: Omit<PythonCellMemberLaunch, "launchIndex" | "command"> = {
          kind: "cell-member",
          processId: processIdentity(routeId, "cell-member", stage, memberId, rank),
          routeId,
          phases: ["prefill", "decode"],
          logicalPlanIds: [...planIds],
          stageId: stage.stageId,
          stageIndex: stage.index,
          layerStart: stage.layerStart,
          layerEnd: stage.layerEnd,
          totalLayers: manifest.totalLayers,
          codec,
          anchor: {
            memberId,
            endpoint: { ...member.endpoint },
          },
          members: structuredClone(stage.members),
          rank,
          fixturePath: external.rankFixturePaths[rank]!,
          pipelineSnapshotIdentity,
          collectiveBackend: stage.execution.collectiveBackend,
          computeDtype: stage.execution.computeDtype,
          device: stage.execution.rankDevices[rank]!,
          cellAnchor: {
            memberId: stage.anchor.memberId,
            controlEndpoint: {
              host: external.controlAdvertiseHost,
              port: external.controlPort,
            },
          },
          worldSize: stage.execution.worldSize,
          operationTimeoutSeconds: stage.execution.operationTimeoutSeconds,
          startupTimeoutSeconds: external.startupTimeoutSeconds,
        };
        launchOrder.push({
          ...memberPartial,
          launchIndex: launchOrder.length,
          command: {
            executable: configuration.pythonExecutable,
            args: renderCellMemberArguments(memberPartial, configuration),
          },
        });
      }
    }
    const processId = processIdentity(routeId, "remote-stage", stage);
    const partial: Omit<PythonRemoteStageLaunch, "launchIndex" | "command"> = {
      kind: "remote-stage",
      processId,
      routeId,
      phases: ["prefill", "decode"],
      logicalPlanIds: [...planIds],
      stageId: stage.stageId,
      stageIndex: stage.index,
      layerStart: stage.layerStart,
      layerEnd: stage.layerEnd,
      totalLayers: manifest.totalLayers,
      codec,
      anchor: structuredClone(stage.anchor),
      members: structuredClone(stage.members),
      downstream,
      returnEndpoint: { ...configuration.returnEndpoint },
      cell: stage.execution ? structuredClone(stage.execution) : null,
    };
    launchOrder.push({
      ...partial,
      launchIndex: launchOrder.length,
      command: {
        executable: configuration.pythonExecutable,
        args: renderRemoteStageArguments(partial, configuration),
      },
    });
  }

  const rootStage = stages[0]!;
  const firstRemoteStage = downstreamStage(stages[1]!);
  const rootProcessId = processIdentity(routeId, "root-engine", rootStage);
  const rootPartial: Omit<PythonRootEngineLaunch, "launchIndex" | "command"> = {
    kind: "root-engine",
    processId: rootProcessId,
    routeId,
    phases: ["prefill", "decode"],
    logicalPlanIds: [...planIds],
    stageId: rootStage.stageId,
    stageIndex: 0,
    layerStart: rootStage.layerStart,
    layerEnd: rootStage.layerEnd,
    totalLayers: manifest.totalLayers,
    codec,
    anchor: structuredClone(rootStage.anchor),
    members: structuredClone(rootStage.members),
    boundaries,
    firstRemoteStage,
    apiEndpoint: { ...configuration.apiEndpoint },
    returnBindHost: configuration.returnBindHost,
    returnEndpoint: { ...configuration.returnEndpoint },
    prefill: structuredClone(phaseSettings[0]),
    decode: structuredClone(phaseSettings[1]),
  };
  launchOrder.push({
    ...rootPartial,
    launchIndex: launchOrder.length,
    command: {
      executable: configuration.pythonExecutable,
      args: renderRootEngineArguments(rootPartial, configuration),
    },
  });

  const remoteStageProcessIds = launchOrder
    .filter((entry): entry is PythonRemoteStageLaunch => entry.kind === "remote-stage")
    .map((entry) => entry.processId);
  const cellMemberProcessIds = launchOrder
    .filter((entry): entry is PythonCellMemberLaunch => entry.kind === "cell-member")
    .map((entry) => entry.processId);
  const route: PythonPipelineLaunchRoute = {
    routeId,
    phases: ["prefill", "decode"],
    phaseSettings,
    codec,
    logicalStageIds: stages.map((stage) => stage.stageId),
    downstreamFirstProcessIds: launchOrder.map((entry) => entry.processId),
    cellMemberProcessIds,
    remoteStageProcessIds,
    rootProcessId,
  };
  const withoutIdentity: Omit<PythonPipelineLaunchDescription, "launchId"> = {
    schema: "gdlp-python-launch/2",
    sourceProtocol: "gdlp/2",
    sourceManifest: structuredClone(manifest),
    pipelineId: manifest.pipelineId,
    modelIdentity: {
      id: manifest.modelId,
      revision: manifest.modelRevision,
      tokenizerId: manifest.tokenizerId,
    },
    runtimeModel: structuredClone(configuration.runtimeModel),
    totalLayers: manifest.totalLayers,
    configuration: structuredClone(configuration),
    route,
    launchOrder,
  };
  return {
    ...withoutIdentity,
    launchId: digest(canonicalJson(withoutIdentity), 24),
  };
}

function assertSharedExecutableRoute(
  manifest: RuntimePipelineManifestV2,
  prefill: RuntimePrefillPlanManifest,
  decode: RuntimeDecodePlanManifest,
): void {
  if (hasDifferingPhaseCellProfiles(prefill.stages, decode.stages)) {
    throw new Error("python_server_does_not_execute_phase_specific_cell_profiles");
  }
  if (!samePhysicalTopology(prefill.stages, decode.stages)) {
    throw new Error("python_server_requires_shared_prefill_decode_route");
  }
  if (prefill.activationCodec !== decode.activationCodec) {
    throw new Error("python_shared_route_requires_one_codec");
  }
  if (manifest.kvTransition.mode !== "in-place") {
    throw new Error("python_server_requires_in_place_kv");
  }
  if (prefill.stages.length < 2) {
    throw new Error("python_distributed_runtime_requires_two_stages");
  }
  for (const [phase, plan] of [
    ["prefill", prefill],
    ["decode", decode],
  ] as const) {
    for (const stage of plan.stages) {
      if (stage.index === 0 && stage.execution !== undefined) {
        throw new Error(`python_cell_stage_cannot_be_root:${phase}:${stage.stageId}`);
      }
      if (stage.last && stage.execution !== undefined) {
        throw new Error(`python_cell_stage_cannot_be_final:${phase}:${stage.stageId}`);
      }
      if (stage.members.length !== 1 && stage.execution === undefined) {
        throw new Error(`python_stage_runtime_requires_single_member:${phase}:${stage.stageId}`);
      }
      if (stage.members.length === 1 && stage.execution !== undefined) {
        throw new Error(`python_cell_stage_requires_multiple_members:${phase}:${stage.stageId}`);
      }
      if (!stage.members.some((member) => member.nodeId === stage.anchor.memberId)) {
        throw new Error(`python_stage_anchor_is_not_physical_member:${phase}:${stage.stageId}`);
      }
    }
  }
}

function hasDifferingPhaseCellProfiles(
  prefill: RuntimeVirtualStageManifest[],
  decode: RuntimeVirtualStageManifest[],
): boolean {
  const count = Math.max(prefill.length, decode.length);
  for (let index = 0; index < count; index += 1) {
    const prefillExecution = prefill[index]?.execution;
    const decodeExecution = decode[index]?.execution;
    if (
      (prefillExecution !== undefined || decodeExecution !== undefined) &&
      canonicalJson(prefillExecution ?? null) !== canonicalJson(decodeExecution ?? null)
    ) {
      return true;
    }
  }
  return false;
}

function renderRemoteStageArguments(
  launch: Omit<PythonRemoteStageLaunch, "launchIndex" | "command">,
  configuration: PythonLaunchConfiguration,
): string[] {
  const args = pythonModulePrefix(configuration.stageModule);
  appendModelArguments(args, configuration.runtimeModel);
  args.push(
    "--layer-start",
    String(launch.layerStart),
    "--layer-end",
    String(launch.layerEnd),
    "--total-layers",
    String(launch.totalLayers),
    "--threads",
    String(configuration.threadsPerStage),
    "--listen-host",
    launch.anchor.endpoint.host,
    "--listen-port",
    String(launch.anchor.endpoint.port),
  );
  if (launch.downstream) {
    args.push(
      "--next-host",
      launch.downstream.endpoint.host,
      "--next-port",
      String(launch.downstream.endpoint.port),
      "--next-layer-end",
      String(launch.downstream.layerEnd),
    );
  }
  if (launch.cell) {
    args.push(
      "--cell-fixture",
      launch.cell.fixture.path,
      "--cell-world-size",
      String(launch.cell.worldSize),
      "--cell-manifest-sha256",
      launch.cell.fixture.manifestSha256,
      "--cell-collective-backend",
      launch.cell.collectiveBackend,
      "--cell-compute-dtype",
      launch.cell.computeDtype,
      "--cell-operation-timeout-seconds",
      finiteNumber(launch.cell.operationTimeoutSeconds),
    );
    for (const device of launch.cell.rankDevices) {
      args.push("--cell-device", device);
    }
    if (launch.cell.fixture.location === "member-local") {
      const external = launch.cell.external!;
      const reservedPorts = new Set([
        launch.anchor.endpoint.port,
        launch.returnEndpoint.port,
        ...(launch.downstream ? [launch.downstream.endpoint.port] : []),
      ]);
      if (
        reservedPorts.has(external.controlPort) ||
        reservedPorts.has(external.distributedPort)
      ) {
        throw new Error("python_external_cell_ports_conflict_with_pipeline");
      }
      args.push(
        "--cell-mode",
        "external",
        "--cell-control-host",
        external.controlBindHost,
        "--cell-control-port",
        String(external.controlPort),
        "--cell-control-advertise-host",
        external.controlAdvertiseHost,
        "--cell-distributed-advertise-host",
        external.distributedAdvertiseHost,
        "--cell-distributed-port",
        String(external.distributedPort),
        "--cell-startup-timeout-seconds",
        finiteNumber(external.startupTimeoutSeconds),
      );
    }
  }
  args.push(
    "--return-host",
    configuration.returnEndpoint.host,
    "--return-port",
    String(configuration.returnEndpoint.port),
    "--codec",
    launch.codec,
    "--connect-timeout-seconds",
    finiteNumber(configuration.connectTimeoutSeconds),
  );
  return args;
}

function renderCellMemberArguments(
  launch: Omit<PythonCellMemberLaunch, "launchIndex" | "command">,
  configuration: PythonLaunchConfiguration,
): string[] {
  return [
    ...pythonModulePrefix("distributed_runtime.cell_member_cli"),
    "--fixture",
    launch.fixturePath,
    "--rank",
    String(launch.rank),
    "--world-size",
    String(launch.worldSize),
    "--pipeline-id",
    launch.pipelineSnapshotIdentity,
    "--cell-collective-backend",
    launch.collectiveBackend,
    "--cell-compute-dtype",
    launch.computeDtype,
    "--cell-device",
    launch.device,
    "--layer-start",
    String(launch.layerStart),
    "--layer-end",
    String(launch.layerEnd),
    "--control-host",
    launch.cellAnchor.controlEndpoint.host,
    "--control-port",
    String(launch.cellAnchor.controlEndpoint.port),
    "--threads",
    String(configuration.threadsPerStage),
    "--connect-timeout-seconds",
    finiteNumber(launch.startupTimeoutSeconds),
    "--operation-timeout-seconds",
    finiteNumber(launch.operationTimeoutSeconds),
  ];
}

function renderRootEngineArguments(
  launch: Omit<PythonRootEngineLaunch, "launchIndex" | "command">,
  configuration: PythonLaunchConfiguration,
): string[] {
  const speculation = pythonSpeculation(launch.decode.speculation);
  const maxActiveSequences = Math.max(
    launch.prefill.microBatchSize,
    launch.decode.microBatchSize,
  );
  if (configuration.maxPendingRequests < maxActiveSequences) {
    throw new Error("python_max_pending_requests_is_too_small");
  }
  const args = pythonModulePrefix(configuration.serverModule);
  appendModelArguments(args, configuration.runtimeModel);
  args.push(
    "--public-model-name",
    configuration.publicModelName,
    "--host",
    configuration.apiEndpoint.host,
    "--port",
    String(configuration.apiEndpoint.port),
    "--boundaries",
    launch.boundaries.join(","),
    "--codec",
    launch.codec,
    "--threads-per-stage",
    String(configuration.threadsPerStage),
    "--max-batch-size",
    String(launch.decode.microBatchSize),
    "--max-active-sequences",
    String(maxActiveSequences),
    "--max-pending-requests",
    String(configuration.maxPendingRequests),
    "--batch-window-ms",
    finiteNumber(configuration.batchWindowMs),
    "--prefill-chunk-tokens",
    String(launch.prefill.chunkTokens),
    "--speculation",
    speculation.provider,
    "--speculative-max-draft-tokens",
    String(speculation.maxDraftTokens),
    "--speculation-minimum-speedup",
    finiteNumber(configuration.speculationMinimumSpeedup),
    "--max-output-tokens",
    String(configuration.maxOutputTokens),
    "--first-stage-host",
    launch.firstRemoteStage.endpoint.host,
    "--first-stage-port",
    String(launch.firstRemoteStage.endpoint.port),
    "--return-bind-host",
    configuration.returnBindHost,
    "--return-advertise-host",
    configuration.returnEndpoint.host,
    "--return-port",
    String(configuration.returnEndpoint.port),
    "--startup-timeout-seconds",
    finiteNumber(configuration.connectTimeoutSeconds),
    "--socket-timeout-seconds",
    finiteNumber(configuration.connectTimeoutSeconds),
  );
  return args;
}

function pythonSpeculation(policy: RuntimeSpeculationPolicy): PythonSpeculationArguments {
  const selected = policy.strategies.find(
    (strategy) => strategy.id === policy.defaultStrategyId,
  );
  if (!selected) throw new Error("python_speculation_default_is_missing");
  if (policy.mode === "disabled" || selected.kind === "autoregressive") {
    return { provider: "off", maxDraftTokens: 1 };
  }
  if (selected.kind !== "ngram") {
    throw new Error(`python_speculation_strategy_not_supported:${selected.kind}`);
  }
  if (selected.maxDraftTokens > 16) {
    throw new Error("python_speculation_draft_length_exceeds_runtime_limit");
  }
  return { provider: "ngram", maxDraftTokens: selected.maxDraftTokens };
}

function normalizeConfiguration(
  manifest: RuntimePipelineManifestV2,
  value: PythonLaunchCompilerOptions,
  requireNormalized = false,
): PythonLaunchConfiguration {
  if (!isRecord(value)) throw new Error("python_launch_options_must_be_an_object");
  validateEndpoint(value.apiEndpoint, "python_api_endpoint_is_invalid");
  validateEndpoint(value.returnEndpoint, "python_return_endpoint_is_invalid");
  const returnBindHost = safeString(
    value.returnBindHost,
    "python_return_bind_host_is_invalid",
  );
  if (value.apiEndpoint.port === value.returnEndpoint.port) {
    throw new Error("python_api_and_return_ports_conflict");
  }
  const runtimeModelValue = value.runtimeModel;
  if (runtimeModelValue !== undefined && !isRecord(runtimeModelValue)) {
    throw new Error("python_runtime_model_is_invalid");
  }
  const source = safeString(
    runtimeModelValue?.source ?? manifest.modelId,
    "python_runtime_model_source_is_invalid",
  );
  const revisionValue = runtimeModelValue?.revision;
  const revision =
    revisionValue === undefined || revisionValue === null
      ? null
      : safeString(revisionValue, "python_runtime_model_revision_is_invalid");
  const derivedSnapshotIdentity = deriveHubSnapshotIdentity(source, revision);
  const explicitSnapshotIdentity = runtimeModelValue?.snapshotIdentity;
  const snapshotIdentity =
    explicitSnapshotIdentity === undefined
      ? derivedSnapshotIdentity
      : uint64String(
          explicitSnapshotIdentity,
          "python_runtime_model_snapshot_identity_is_invalid",
        );
  if (
    snapshotIdentity !== undefined &&
    derivedSnapshotIdentity !== undefined &&
    snapshotIdentity !== derivedSnapshotIdentity
  ) {
    throw new Error("python_runtime_model_snapshot_identity_mismatch");
  }
  const normalized: PythonLaunchConfiguration = {
    apiEndpoint: { ...value.apiEndpoint },
    returnEndpoint: { ...value.returnEndpoint },
    returnBindHost,
    runtimeModel: {
      source,
      revision,
      ...(snapshotIdentity ? { snapshotIdentity } : {}),
    },
    publicModelName: safeString(
      value.publicModelName ?? manifest.modelId,
      "python_public_model_name_is_invalid",
    ),
    pythonExecutable: safeString(
      value.pythonExecutable ?? "python",
      "python_executable_is_invalid",
    ),
    stageModule: pythonModule(
      value.stageModule ?? "distributed_runtime.stage_cli",
      "python_stage_module_is_invalid",
    ),
    serverModule: pythonModule(
      value.serverModule ?? "distributed_runtime.server",
      "python_server_module_is_invalid",
    ),
    threadsPerStage: boundedInteger(
      value.threadsPerStage ?? 1,
      1,
      1_024,
      "python_threads_per_stage_is_invalid",
    ),
    connectTimeoutSeconds: positiveFinite(
      value.connectTimeoutSeconds ?? 180,
      "python_connect_timeout_is_invalid",
    ),
    batchWindowMs: nonNegativeFinite(
      value.batchWindowMs ?? 2,
      "python_batch_window_is_invalid",
    ),
    maxPendingRequests: boundedInteger(
      value.maxPendingRequests ?? 128,
      1,
      1_000_000,
      "python_max_pending_requests_is_invalid",
    ),
    maxOutputTokens: boundedInteger(
      value.maxOutputTokens ?? 512,
      1,
      1_000_000,
      "python_max_output_tokens_is_invalid",
    ),
    speculationMinimumSpeedup: atLeastOneFinite(
      value.speculationMinimumSpeedup ?? 1.05,
      "python_speculation_speedup_is_invalid",
    ),
  };
  if (requireNormalized && canonicalJson(value) !== canonicalJson(normalized)) {
    throw new Error("python_launch_configuration_is_not_normalized");
  }
  return normalized;
}

function prefillSettings(plan: RuntimePrefillPlanManifest): PythonPrefillLaunchSettings {
  return {
    phase: "prefill",
    planId: plan.planId,
    activationCodec: plan.activationCodec,
    microBatchSize: plan.microBatchSize,
    chunkTokens: plan.chunkTokens,
  };
}

function decodeSettings(plan: RuntimeDecodePlanManifest): PythonDecodeLaunchSettings {
  return {
    phase: "decode",
    planId: plan.planId,
    activationCodec: plan.activationCodec,
    microBatchSize: plan.microBatchSize,
    directTokenReturnStage: plan.directTokenReturnStage,
    speculation: structuredClone(plan.speculation),
  };
}

function samePhysicalTopology(
  left: RuntimeVirtualStageManifest[],
  right: RuntimeVirtualStageManifest[],
): boolean {
  if (left.length !== right.length) return false;
  return left.every((stage, index) => {
    const other = right[index];
    return (
      other !== undefined &&
      stage.stageId === other.stageId &&
      stage.index === other.index &&
      stage.layerStart === other.layerStart &&
      stage.layerEnd === other.layerEnd &&
      canonicalJson(stage.anchor) === canonicalJson(other.anchor) &&
      canonicalJson(
        stage.members.map((member) => ({
          nodeId: member.nodeId,
          endpoint: member.endpoint,
          backend: member.backend,
          capabilities: member.capabilities,
        })),
      ) ===
        canonicalJson(
          other.members.map((member) => ({
            nodeId: member.nodeId,
            endpoint: member.endpoint,
            backend: member.backend,
            capabilities: member.capabilities,
          })),
        ) &&
      canonicalJson(stage.execution) === canonicalJson(other.execution)
    );
  });
}

function downstreamStage(stage: RuntimeVirtualStageManifest): PythonDownstreamStage {
  return {
    stageId: stage.stageId,
    stageIndex: stage.index,
    layerEnd: stage.layerEnd,
    anchorMemberId: stage.anchor.memberId,
    endpoint: { ...stage.anchor.endpoint },
  };
}

function routeIdentity(
  pipelineId: string,
  settings: [PythonPrefillLaunchSettings, PythonDecodeLaunchSettings],
  stages: RuntimeVirtualStageManifest[],
  codec: RuntimeActivationCodec,
): string {
  return `route-${digest(
    canonicalJson({
      pipelineId,
      settings,
      codec,
      stages: stages.map((stage) => ({
        stageId: stage.stageId,
        index: stage.index,
        layers: [stage.layerStart, stage.layerEnd],
        anchor: stage.anchor,
        members: stage.members.map((member) => ({
          nodeId: member.nodeId,
          endpoint: member.endpoint,
          backend: member.backend,
          capabilities: member.capabilities,
        })),
        execution: stage.execution,
      })),
    }),
    20,
  )}`;
}

function processIdentity(
  routeId: string,
  kind: PythonLaunchProcess["kind"],
  stage: RuntimeVirtualStageManifest,
  memberId?: string,
  rank?: number,
): string {
  const prefix =
    kind === "root-engine" ? "root" : kind === "cell-member" ? "cell" : "stage";
  return `${prefix}-${digest(
    canonicalJson({
      routeId,
      kind,
      stageId: stage.stageId,
      index: stage.index,
      layers: [stage.layerStart, stage.layerEnd],
      anchor: stage.anchor,
      execution: stage.execution,
      memberId: memberId ?? null,
      rank: rank ?? null,
    }),
    20,
  )}`;
}

function pythonModulePrefix(module: string): string[] {
  // Readiness is observed over stdout/stderr pipes. Force unbuffered Python so
  // a service cannot be healthy yet remain invisible to the supervisor until
  // its stdio buffer fills or the process exits.
  return ["-u", "-m", module];
}

function appendModelArguments(args: string[], model: PythonRuntimeModelSource): void {
  args.push("--model", model.source);
  if (model.revision !== null) args.push("--revision", model.revision);
}

function deriveHubSnapshotIdentity(
  source: string,
  revision: string | null,
): string | undefined {
  const revisionCandidate = revision?.trim().toLowerCase();
  const sourceParts = source.split(/[\\/]+/).filter(Boolean);
  const pathCandidate =
    sourceParts.length >= 2 && sourceParts.at(-2)?.toLowerCase() === "snapshots"
      ? sourceParts.at(-1)!.toLowerCase()
      : undefined;
  const commit = [revisionCandidate, pathCandidate].find(
    (candidate) =>
      candidate !== undefined &&
      candidate.length >= 32 &&
      /^[0-9a-f]+$/.test(candidate),
  );
  if (!commit) return undefined;
  const bytes = createHash("sha256")
    .update("gdlp-hub-snapshot-v1\0")
    .update(commit, "ascii")
    .digest()
    .subarray(0, 8);
  let identity = 0n;
  for (const byte of bytes) identity = (identity << 8n) | BigInt(byte);
  return identity.toString();
}

function uint64String(value: unknown, error: string): string {
  if (typeof value !== "string" || !/^(?:0|[1-9]\d*)$/.test(value)) {
    throw new Error(error);
  }
  const parsed = BigInt(value);
  if (parsed > 18_446_744_073_709_551_615n) throw new Error(error);
  return parsed.toString();
}

function validateEndpoint(value: unknown, error: string): asserts value is RuntimeEndpoint {
  if (!isRecord(value) || typeof value.host !== "string" || !value.host.trim()) {
    throw new Error(error);
  }
  if (!Number.isInteger(value.port) || (value.port as number) < 1 || (value.port as number) > 65_535) {
    throw new Error(error);
  }
}

function endpointKey(endpoint: RuntimeEndpoint): string {
  return `${endpoint.host.trim().toLowerCase()}\u0000${endpoint.port}`;
}

function pythonModule(value: unknown, error: string): string {
  const result = safeString(value, error);
  if (!/^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/.test(result)) throw new Error(error);
  return result;
}

function safeString(value: unknown, error: string): string {
  if (
    typeof value !== "string" ||
    !value.trim() ||
    value.includes("\u0000") ||
    value.includes("\r") ||
    value.includes("\n")
  ) {
    throw new Error(error);
  }
  return value;
}

function boundedInteger(value: unknown, min: number, max: number, error: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < min || (value as number) > max) {
    throw new Error(error);
  }
  return value as number;
}

function positiveFinite(value: unknown, error: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(error);
  }
  return value;
}

function nonNegativeFinite(value: unknown, error: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error(error);
  }
  return value;
}

function atLeastOneFinite(value: unknown, error: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 1) {
    throw new Error(error);
  }
  return value;
}

function finiteNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6)));
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (!isRecord(value)) return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonicalValue(value[key])]),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function digest(value: string, length: number): string {
  return createHash("sha256").update(value).digest("hex").slice(0, length);
}
