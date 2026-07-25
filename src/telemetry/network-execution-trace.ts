import { z } from "zod";
import type {
  ModelDeployment,
  NetworkExecutionBoundaryTrace,
  NetworkExecutionTrace,
  NetworkExecutionTraceStage,
  ScheduledRoute,
} from "../contracts/types.js";
import type { StoredWorker } from "../storage/store.js";
import type { RuntimeTransportSnapshot } from "../coordinator/worker-hub.js";

export const NETWORK_EXECUTION_TRACE_SCHEMA =
  "mycellios-network-execution-trace/1" as const;

const identifier = z.string().min(1).max(512);
const nullableTimestamp = z.number().int().nonnegative().nullable();
const nullableMeasurement = z.number().nonnegative().finite().nullable();
const executionBackend = z.enum([
  "cpu",
  "cuda",
  "rocm",
  "directml",
  "mps",
  "xpu",
  "vulkan",
  "webgpu",
]);

const traceStageSchema = z.object({
  routeStageIndex: z.number().int().nonnegative(),
  stageIndex: z.number().int().nonnegative(),
  nodeId: identifier.nullable(),
  workerId: identifier.nullable(),
  deploymentId: identifier,
  deploymentOwnerWorkerId: identifier,
  modelDigest: identifier,
  layerStart: z.number().int().nonnegative().nullable(),
  layerEnd: z.number().int().positive().nullable(),
  deviceType: z.enum(["cpu", "gpu"]),
  backend: executionBackend,
  precision: z.string().min(1).max(64),
  deviceName: z.string().min(1).max(256),
  startedAt: nullableTimestamp,
  endedAt: nullableTimestamp,
  durationMs: nullableMeasurement,
}).strict().superRefine((stage, context) => {
  if ((stage.layerStart === null) !== (stage.layerEnd === null)) {
    context.addIssue({
      code: "custom",
      message: "layer range endpoints must both be observed or both be null",
      path: ["layerEnd"],
    });
  }
  if (
    stage.layerStart !== null
    && stage.layerEnd !== null
    && stage.layerEnd <= stage.layerStart
  ) {
    context.addIssue({
      code: "custom",
      message: "layerEnd must be greater than layerStart",
      path: ["layerEnd"],
    });
  }
  const timing = [stage.startedAt, stage.endedAt, stage.durationMs];
  if (timing.some((value) => value !== null) && timing.some((value) => value === null)) {
    context.addIssue({
      code: "custom",
      message: "stage timing must be complete or entirely unobserved",
      path: ["durationMs"],
    });
  }
  if (
    stage.startedAt !== null
    && stage.endedAt !== null
    && stage.durationMs !== null
    && (
      stage.endedAt < stage.startedAt
      || stage.durationMs !== stage.endedAt - stage.startedAt
    )
  ) {
    context.addIssue({
      code: "custom",
      message: "stage timing is inconsistent",
      path: ["durationMs"],
    });
  }
});

const boundarySchema = z.object({
  boundaryIndex: z.number().int().nonnegative(),
  fromStageIndex: z.number().int().nonnegative(),
  toStageIndex: z.number().int().nonnegative(),
  sourceNodeId: identifier.nullable(),
  destinationNodeId: identifier.nullable(),
  physicalBoundary: z.boolean().nullable(),
  transport: z.enum(["direct", "relay", "local", "unobserved"]),
  streamId: identifier.nullable(),
  bytesSourceToDestination: z.number().int().nonnegative().nullable(),
  bytesDestinationToSource: z.number().int().nonnegative().nullable(),
  countersExclusive: z.boolean().nullable(),
  connectRttMs: nullableMeasurement,
  streamCreatedAt: nullableTimestamp,
  streamConnectedAt: nullableTimestamp,
  streamEndedAt: nullableTimestamp,
  observedOverlapMs: nullableMeasurement,
}).strict().superRefine((boundary, context) => {
  if (boundary.transport === "local") {
    if (
      boundary.physicalBoundary !== false
      || boundary.streamId !== null
      || boundary.bytesSourceToDestination !== null
      || boundary.bytesDestinationToSource !== null
      || boundary.countersExclusive !== null
      || boundary.connectRttMs !== null
      || boundary.streamCreatedAt !== null
      || boundary.streamConnectedAt !== null
      || boundary.streamEndedAt !== null
      || boundary.observedOverlapMs !== null
    ) {
      context.addIssue({
        code: "custom",
        message: "local transport cannot claim network measurements",
        path: ["transport"],
      });
    }
  } else if (boundary.transport === "unobserved") {
    if (
      boundary.streamId !== null
      || boundary.bytesSourceToDestination !== null
      || boundary.bytesDestinationToSource !== null
      || boundary.countersExclusive !== null
      || boundary.connectRttMs !== null
      || boundary.streamCreatedAt !== null
      || boundary.streamConnectedAt !== null
      || boundary.streamEndedAt !== null
      || boundary.observedOverlapMs !== null
    ) {
      context.addIssue({
        code: "custom",
        message: "unobserved transport cannot carry measured values",
        path: ["transport"],
      });
    }
  } else if (
    boundary.physicalBoundary !== true
    || boundary.streamId === null
    || boundary.streamCreatedAt === null
    || boundary.observedOverlapMs === null
    || boundary.countersExclusive === null
  ) {
    context.addIssue({
      code: "custom",
      message: "network transport requires an observed physical stream",
      path: ["transport"],
    });
  }
  const byteReadings = [
    boundary.bytesSourceToDestination,
    boundary.bytesDestinationToSource,
  ];
  if (boundary.countersExclusive === true && byteReadings.some((value) => value === null)) {
    context.addIssue({
      code: "custom",
      message: "exclusive counters require both byte readings",
      path: ["countersExclusive"],
    });
  }
  if (boundary.countersExclusive !== true && byteReadings.some((value) => value !== null)) {
    context.addIssue({
      code: "custom",
      message: "shared or unknown counters cannot be attributed to a request",
      path: ["countersExclusive"],
    });
  }
});

export const networkExecutionTraceSchema = z.object({
  schema: z.literal(NETWORK_EXECUTION_TRACE_SCHEMA),
  jobId: identifier,
  attempt: z.number().int().positive(),
  observedFrom: z.number().int().nonnegative(),
  observedUntil: z.number().int().nonnegative(),
  durationMs: z.number().int().nonnegative(),
  routeClass: z.enum(["replica", "pipeline"]),
  affinityHit: z.boolean(),
  selectedRoute: z.array(z.object({
    routeStageIndex: z.number().int().nonnegative(),
    workerId: identifier,
    deploymentId: identifier,
    modelDigest: identifier,
    stageIndex: z.number().int().nonnegative(),
  }).strict()).min(1).max(64),
  stages: z.array(traceStageSchema).max(256),
  physicalBoundaryCount: z.number().int().nonnegative().nullable(),
  boundaries: z.array(boundarySchema).max(255),
}).strict().superRefine((trace, context) => {
  if (
    trace.observedUntil < trace.observedFrom
    || trace.durationMs !== trace.observedUntil - trace.observedFrom
  ) {
    context.addIssue({
      code: "custom",
      message: "request observation interval is inconsistent",
      path: ["durationMs"],
    });
  }
  if (trace.boundaries.length !== Math.max(0, trace.stages.length - 1)) {
    context.addIssue({
      code: "custom",
      message: "boundary count must match the observed stage sequence",
      path: ["boundaries"],
    });
  }
  for (const [index, selected] of trace.selectedRoute.entries()) {
    if (selected.routeStageIndex !== index) {
      context.addIssue({
        code: "custom",
        message: "selected route indexes must be contiguous and ordered",
        path: ["selectedRoute", index, "routeStageIndex"],
      });
    }
  }
  for (const [index, boundary] of trace.boundaries.entries()) {
    const from = trace.stages[index]!;
    const to = trace.stages[index + 1]!;
    if (
      boundary.boundaryIndex !== index
      || boundary.fromStageIndex !== from.stageIndex
      || boundary.toStageIndex !== to.stageIndex
      || boundary.sourceNodeId !== from.nodeId
      || boundary.destinationNodeId !== to.nodeId
    ) {
      context.addIssue({
        code: "custom",
        message: "boundary does not match its adjacent observed stages",
        path: ["boundaries", index],
      });
    }
    const derivedPhysical = from.nodeId === null || to.nodeId === null
      ? null
      : from.nodeId !== to.nodeId;
    if (boundary.physicalBoundary !== derivedPhysical) {
      context.addIssue({
        code: "custom",
        message: "physical boundary flag does not match the observed node identities",
        path: ["boundaries", index, "physicalBoundary"],
      });
    }
  }
  for (const [index, stage] of trace.stages.entries()) {
    const selected = trace.selectedRoute[stage.routeStageIndex];
    if (
      !selected
      || stage.deploymentId !== selected.deploymentId
      || stage.deploymentOwnerWorkerId !== selected.workerId
      || stage.modelDigest !== selected.modelDigest
    ) {
      context.addIssue({
        code: "custom",
        message: "observed stage is not backed by its selected deployment",
        path: ["stages", index],
      });
    }
  }
  const knownPhysicalBoundaries = trace.stages.length > 0 && trace.boundaries.every(
    (boundary) => boundary.physicalBoundary !== null,
  );
  const derivedPhysicalBoundaryCount = trace.boundaries.filter(
    (boundary) => boundary.physicalBoundary === true,
  ).length;
  if (
    trace.physicalBoundaryCount !== (knownPhysicalBoundaries
      ? derivedPhysicalBoundaryCount
      : null)
  ) {
    context.addIssue({
      code: "custom",
      message: "physical boundary count is inconsistent",
      path: ["physicalBoundaryCount"],
    });
  }
});

export interface BuildNetworkExecutionTraceInput {
  jobId: string;
  attempt: number;
  route: ScheduledRoute;
  workers: readonly StoredWorker[];
  observedFrom: number;
  observedUntil: number;
  startTransports: readonly RuntimeTransportSnapshot[];
  endTransports: readonly RuntimeTransportSnapshot[];
  /** Directed `source\0destination` keys used by another overlapping job. */
  contendedBoundaryKeys?: ReadonlySet<string>;
}

export function buildNetworkExecutionTrace(
  input: BuildNetworkExecutionTraceInput,
): NetworkExecutionTrace {
  const observedFrom = boundedTimestamp(input.observedFrom);
  const observedUntil = boundedTimestamp(input.observedUntil);
  if (observedUntil < observedFrom) throw new Error("network_trace_interval_is_invalid");
  const stages = resolveTraceStages(input.route, input.workers);
  const ambiguousDeploymentBoundaryKeys = deploymentAmbiguousBoundaryKeys(
    input.route,
    input.workers,
  );
  const boundaries = stages.slice(1).map((stage, index) =>
    buildBoundary({
      index,
      from: stages[index]!,
      to: stage,
      observedFrom,
      observedUntil,
      startTransports: input.startTransports,
      endTransports: input.endTransports,
      contendedBoundaryKeys: input.contendedBoundaryKeys ?? new Set(),
      ambiguousDeploymentBoundaryKeys,
    })
  );
  const physicalBoundaryCount = stages.length > 0 && boundaries.every(
    (boundary) => boundary.physicalBoundary !== null,
  )
    ? boundaries.filter((boundary) => boundary.physicalBoundary).length
    : null;
  return networkExecutionTraceSchema.parse({
    schema: NETWORK_EXECUTION_TRACE_SCHEMA,
    jobId: input.jobId,
    attempt: input.attempt,
    observedFrom,
    observedUntil,
    durationMs: observedUntil - observedFrom,
    routeClass: input.route.routeClass,
    affinityHit: input.route.affinityHit,
    selectedRoute: input.route.stages.map((stage, routeStageIndex) => ({
      routeStageIndex,
      workerId: stage.workerId,
      deploymentId: stage.deploymentId,
      modelDigest: stage.modelDigest,
      stageIndex: stage.stageIndex,
    })),
    stages,
    physicalBoundaryCount,
    boundaries,
  });
}

export function parseNetworkExecutionTrace(value: unknown): NetworkExecutionTrace | null {
  const parsed = networkExecutionTraceSchema.safeParse(value);
  return parsed.success ? parsed.data : null;
}

export function networkTraceBoundaryKeys(
  route: ScheduledRoute,
  workers: readonly StoredWorker[],
): Set<string> {
  const stages = resolveTraceStages(route, workers);
  return new Set(stages.slice(1).flatMap((stage, index) => {
    const sourceNodeId = stages[index]!.nodeId;
    return sourceNodeId && stage.nodeId && sourceNodeId !== stage.nodeId
      ? [directedBoundaryKey(sourceNodeId, stage.nodeId)]
      : [];
  }));
}

export function directedBoundaryKey(sourceNodeId: string, destinationNodeId: string): string {
  return `${sourceNodeId}\u0000${destinationNodeId}`;
}

function resolveTraceStages(
  route: ScheduledRoute,
  workers: readonly StoredWorker[],
): NetworkExecutionTraceStage[] {
  const workersById = new Map(workers.map((worker) => [worker.id, worker]));
  const workersByNodeId = new Map<string, StoredWorker[]>();
  for (const worker of workers) {
    const nodeId = worker.capabilities.distributedExecutor?.nodeId;
    if (!nodeId) continue;
    const entries = workersByNodeId.get(nodeId) ?? [];
    entries.push(worker);
    workersByNodeId.set(nodeId, entries);
  }
  const result: NetworkExecutionTraceStage[] = [];
  for (const [routeStageIndex, selected] of route.stages.entries()) {
    const owner = workersById.get(selected.workerId);
    const deployment = owner?.capabilities.deployments.find(
      (candidate) =>
        candidate.deploymentId === selected.deploymentId
        && candidate.modelDigest === selected.modelDigest,
    );
    if (!owner || !deployment?.execution) continue;
    const executionStages = deployment.execution.stages;
    if (executionStages?.length) {
      for (const stage of [...executionStages].sort(
        (left, right) => left.stageIndex - right.stageIndex,
      )) {
        const workerMatches = workersByNodeId.get(stage.nodeId) ?? [];
        result.push({
          routeStageIndex,
          stageIndex: stage.stageIndex,
          nodeId: stage.nodeId,
          workerId: workerMatches.length === 1 ? workerMatches[0]!.id : null,
          deploymentId: deployment.deploymentId,
          deploymentOwnerWorkerId: owner.id,
          modelDigest: deployment.modelDigest,
          layerStart: stage.layerStart,
          layerEnd: stage.layerEnd,
          deviceType: stage.deviceType,
          backend: stage.backend,
          precision: stage.precision,
          deviceName: stage.deviceName,
          startedAt: null,
          endedAt: null,
          durationMs: null,
        });
      }
      continue;
    }
    const ownerNodeId = owner.capabilities.distributedExecutor?.nodeId ?? null;
    if (deployment.execution.deviceType === "mixed") continue;
    result.push(traceStageFromDeployment(
      routeStageIndex,
      selected.stageIndex,
      owner,
      deployment,
      ownerNodeId,
    ));
  }
  return result;
}

function traceStageFromDeployment(
  routeStageIndex: number,
  selectedStageIndex: number,
  owner: StoredWorker,
  deployment: ModelDeployment,
  nodeId: string | null,
): NetworkExecutionTraceStage {
  const execution = deployment.execution!;
  if (execution.deviceType === "mixed") {
    throw new Error("network_trace_mixed_execution_requires_stage_evidence");
  }
  return {
    routeStageIndex,
    stageIndex: deployment.stage?.index ?? selectedStageIndex,
    nodeId,
    workerId: owner.id,
    deploymentId: deployment.deploymentId,
    deploymentOwnerWorkerId: owner.id,
    modelDigest: deployment.modelDigest,
    layerStart: deployment.stage?.layerStart ?? null,
    layerEnd: deployment.stage?.layerEnd ?? null,
    deviceType: execution.deviceType,
    backend: execution.backend,
    precision: execution.precision,
    deviceName: execution.deviceName,
    startedAt: null,
    endedAt: null,
    durationMs: null,
  };
}

function buildBoundary(input: {
  index: number;
  from: NetworkExecutionTraceStage;
  to: NetworkExecutionTraceStage;
  observedFrom: number;
  observedUntil: number;
  startTransports: readonly RuntimeTransportSnapshot[];
  endTransports: readonly RuntimeTransportSnapshot[];
  contendedBoundaryKeys: ReadonlySet<string>;
  ambiguousDeploymentBoundaryKeys: ReadonlySet<string>;
}): NetworkExecutionBoundaryTrace {
  const sourceNodeId = input.from.nodeId;
  const destinationNodeId = input.to.nodeId;
  const base = {
    boundaryIndex: input.index,
    fromStageIndex: input.from.stageIndex,
    toStageIndex: input.to.stageIndex,
    sourceNodeId,
    destinationNodeId,
  };
  if (!sourceNodeId || !destinationNodeId) {
    return unobservedBoundary(base, null);
  }
  if (sourceNodeId === destinationNodeId) {
    return {
      ...base,
      physicalBoundary: false,
      transport: "local",
      streamId: null,
      bytesSourceToDestination: null,
      bytesDestinationToSource: null,
      countersExclusive: null,
      connectRttMs: null,
      streamCreatedAt: null,
      streamConnectedAt: null,
      streamEndedAt: null,
      observedOverlapMs: null,
    };
  }
  const boundaryKey = directedBoundaryKey(sourceNodeId, destinationNodeId);
  if (input.ambiguousDeploymentBoundaryKeys.has(boundaryKey)) {
    return unobservedBoundary(base, true);
  }
  const candidates = input.endTransports.filter((snapshot) =>
    snapshot.sourceNodeId === sourceNodeId
    && snapshot.destinationNodeId === destinationNodeId
    && snapshot.state !== "negotiating"
    && overlaps(snapshot, input.observedFrom, input.observedUntil)
  );
  if (candidates.length !== 1) return unobservedBoundary(base, true);
  const current = candidates[0]!;
  const start = input.startTransports.find(
    (snapshot) => snapshot.streamId === current.streamId,
  );
  const exclusive = !input.contendedBoundaryKeys.has(boundaryKey);
  const countersStartedDuringRequest = current.createdAt >= input.observedFrom;
  const countersComparable = start !== undefined || countersStartedDuringRequest;
  const sourceStart = start?.bytesSourceToDestination ?? 0;
  const destinationStart = start?.bytesDestinationToSource ?? 0;
  const countersMonotonic = (
    current.bytesSourceToDestination >= sourceStart
    && current.bytesDestinationToSource >= destinationStart
  );
  const observedForwardBytes = current.bytesSourceToDestination - sourceStart;
  const observedReturnBytes = current.bytesDestinationToSource - destinationStart;
  // Relay counters are updated synchronously by the coordinator. Direct
  // counters arrive as bounded telemetry; an unchanged zero is not proof that
  // no bytes crossed the peer connection.
  const countersHaveEvidence = current.mode === "relay"
    || observedForwardBytes > 0
    || observedReturnBytes > 0;
  const countersExclusive = (
    exclusive
    && countersComparable
    && countersMonotonic
    && countersHaveEvidence
  );
  const overlapStart = Math.max(
    input.observedFrom,
    current.connectedAt ?? current.createdAt,
  );
  const overlapEnd = Math.min(
    input.observedUntil,
    current.endedAt ?? input.observedUntil,
  );
  return {
    ...base,
    physicalBoundary: true,
    transport: current.mode,
    streamId: current.streamId,
    bytesSourceToDestination: countersExclusive
      ? observedForwardBytes
      : null,
    bytesDestinationToSource: countersExclusive
      ? observedReturnBytes
      : null,
    countersExclusive,
    connectRttMs: finiteOrNull(current.connectRttMs),
    streamCreatedAt: current.createdAt,
    streamConnectedAt: current.connectedAt,
    streamEndedAt: current.endedAt,
    observedOverlapMs: Math.max(0, overlapEnd - overlapStart),
  };
}

function deploymentAmbiguousBoundaryKeys(
  route: ScheduledRoute,
  workers: readonly StoredWorker[],
): Set<string> {
  const selectedDeployments = new Set(
    route.stages.map((stage) => deploymentIdentity(stage.workerId, stage.deploymentId)),
  );
  const ownersByBoundary = new Map<string, Set<string>>();
  for (const worker of workers) {
    for (const deployment of worker.capabilities.deployments) {
      const stages = deployment.execution?.stages;
      if (!stages || stages.length < 2) continue;
      const ordered = [...stages].sort((left, right) => left.stageIndex - right.stageIndex);
      for (let index = 1; index < ordered.length; index += 1) {
        const source = ordered[index - 1]!.nodeId;
        const destination = ordered[index]!.nodeId;
        if (source === destination) continue;
        const key = directedBoundaryKey(source, destination);
        const owners = ownersByBoundary.get(key) ?? new Set<string>();
        owners.add(deploymentIdentity(worker.id, deployment.deploymentId));
        ownersByBoundary.set(key, owners);
      }
    }
  }
  return new Set([...ownersByBoundary].flatMap(([key, owners]) =>
    owners.size !== 1 || !selectedDeployments.has([...owners][0]!)
      ? [key]
      : []
  ));
}

function deploymentIdentity(workerId: string, deploymentId: string): string {
  return `${workerId}\u0000${deploymentId}`;
}

function unobservedBoundary(
  base: Pick<
    NetworkExecutionBoundaryTrace,
    | "boundaryIndex"
    | "fromStageIndex"
    | "toStageIndex"
    | "sourceNodeId"
    | "destinationNodeId"
  >,
  physicalBoundary: boolean | null,
): NetworkExecutionBoundaryTrace {
  return {
    ...base,
    physicalBoundary,
    transport: "unobserved",
    streamId: null,
    bytesSourceToDestination: null,
    bytesDestinationToSource: null,
    countersExclusive: null,
    connectRttMs: null,
    streamCreatedAt: null,
    streamConnectedAt: null,
    streamEndedAt: null,
    observedOverlapMs: null,
  };
}

function overlaps(
  snapshot: RuntimeTransportSnapshot,
  observedFrom: number,
  observedUntil: number,
): boolean {
  const startedAt = snapshot.connectedAt ?? snapshot.createdAt;
  const endedAt = snapshot.endedAt ?? Number.POSITIVE_INFINITY;
  return startedAt <= observedUntil && endedAt >= observedFrom;
}

function finiteOrNull(value: number | null): number | null {
  return value !== null && Number.isFinite(value) && value >= 0 ? value : null;
}

function boundedTimestamp(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error("network_trace_timestamp_is_invalid");
  }
  return value;
}
