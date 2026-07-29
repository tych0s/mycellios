import type { AutoDistributionConfig } from "../distribution/auto-distribute.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";
import type { LaunchAgent } from "../distribution/launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "../distribution/python-launcher.js";
import { WorkerTunnelLaunchAgent } from "../distribution/worker-tunnel-launch-agent.js";
import type { StoredRequestedModel, StoredWorker } from "../storage/store.js";
import { deriveDecodeScales } from "../distribution/node-scale.js";
import type { WorkerHub } from "./worker-hub.js";
import type { DynamicActivationSnapshot } from "./model-activation-manager.js";
import type { RuntimeLinkObservation } from "./runtime-link-observations.js";
import {
  plannerScalesFromCoordinatorEvidence,
  type PlannerPerformanceScales,
} from "../performance/runtime-profile.js";

interface ConnectedExecutor {
  worker: StoredWorker;
  executor: NonNullable<StoredWorker["capabilities"]["distributedExecutor"]>;
}

interface ProfiledConnectedExecutor extends ConnectedExecutor {
  performance: PlannerPerformanceScales;
}

const RUNTIME_LINK_EVIDENCE_TTL_MS = 5 * 60_000;

/**
 * Decode throughput this worker actually measured **for this model**, or null.
 *
 * Only `measured` counts. An estimated or configured number is a guess about
 * hardware, and planning a layer split on a guess is how the planner ended up
 * trusting a constant in the first place.
 *
 * El filtro por modelo no es cosmético: los tokens por segundo dependen del
 * modelo, así que un `Math.max` sobre TODOS los despliegues del worker compara
 * medidas incomparables. Un nodo que aún conserva la medida de un modelo
 * pequeño declara un caudal altísimo, y `deriveDecodeScales` le adjudica más
 * capas del modelo grande que se está planificando. El sesgo va justo en la
 * dirección mala: premia al que midió con la carga más ligera, que es lo
 * contrario de lo que el reparto proporcional pretende.
 */
function measuredDecodeThroughput(worker: StoredWorker, modelName: string): number | null {
  const measured = worker.capabilities.deployments
    .filter((deployment) => deployment.model === modelName)
    .filter((deployment) => deployment.throughputSource === "measured")
    .map((deployment) => deployment.tokensPerSecond)
    .filter((value) => Number.isFinite(value) && value > 0);
  return measured.length === 0 ? null : Math.max(...measured);
}

/**
 * What we assume a link costs when nobody has measured it yet.
 *
 * `coordinatorRttMs` is 0 until the agent's ping probe lands (and stays 0 for
 * agents older than that probe). The previous code applied a `Math.max(0.1, …)`
 * floor, so an unmeasured pair was modelled as a 0.1 ms link — a LAN-grade
 * number that made every placement look free and hid the single largest term
 * in the cost model.
 *
 * The measured fleet median is 65 ms per hop (Exp15, `docs/benchmarks/
 * gpu_cloud-exp15-mapa-latencia-2026-07-25/`), with a worst observed node at
 * 437 ms. Assuming the median when blind is not accurate, but it is the right
 * kind of wrong: an unmeasured link no longer outranks a measured good one.
 */
const UNMEASURED_LINK_LATENCY_MS = 65;
/** Floor for a genuinely measured link, so a 0 never divides downstream. */
const MIN_LINK_LATENCY_MS = 0.1;

/**
 * One-way latency between two executors, estimated from each one's round trip
 * to the coordinator: one way a→coordinator is `rttA/2`, coordinator→b is
 * `rttB/2`, so a→b via the coordinator is `(rttA + rttB) / 2`. Exp16 measured
 * that a direct node→node hop costs 0.96-1.24x a node→relay hop, so this also
 * approximates the direct path.
 */
export function estimateLinkLatencyMs(fromRttMs: number, toRttMs: number): number {
  const measured = [fromRttMs, toRttMs].filter((rtt) => Number.isFinite(rtt) && rtt > 0);
  if (measured.length === 0) return UNMEASURED_LINK_LATENCY_MS;
  // One side measured is better than none: assume the blind side matches it
  // rather than falling back to the fleet median, which would ignore real data.
  const average = measured.reduce((sum, rtt) => sum + rtt, 0) / measured.length;
  return Math.max(MIN_LINK_LATENCY_MS, average);
}

/**
 * Converts live desktop shard executors into the activation topology consumed by
 * DynamicModelActivationManager. Only connected workers count as capacity.
 */
export function buildConnectedExecutorActivationSnapshot(
  baseConfig: AutoDistributionConfig,
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
  runtimeLinkObservations: readonly RuntimeLinkObservation[] = [],
): DynamicActivationSnapshot {
  const executors = connectedExecutors(workers, connectedWorkerIds);
  const capacityNodes = executors.map(({ worker, executor }) => ({
    id: executor.nodeId,
    availableVramMiB: worker.capabilities.gpus.reduce(
      (sum, gpu) => sum + gpu.freeOfferedVramMb,
      0,
    ),
  }));
  const profiledExecutors = executors.flatMap((entry) => {
    const performance = eligiblePlannerPerformance(entry);
    return performance ? [{ ...entry, performance }] : [];
  });
  const profiledNodeIds = new Set(
    profiledExecutors.map(({ executor }) => executor.nodeId),
  );
  if (profiledExecutors.length < 2) {
    return {
      capacityNodes,
      config: null,
      readinessDetails: executors.map(({ executor }) =>
        profiledNodeIds.has(executor.nodeId)
          ? `${executor.nodeId}: runtime performance verified.`
          : `${executor.nodeId}: connected; waiting for a verified runtime performance profile.`
      ),
    };
  }

  // El SEGUNDO cero del planificador. `coordinatorRttMs` ya se mide, pero
  // `decodeScale` seguía fijado a 1 aquí, así que `ProportionalComputePlanner`
  // dividía por un vector de unos y **el reparto proporcional degeneraba a
  // reparto igual**: el planificador no podía distinguir una 4090 de una
  // 1050 Ti. El planificador proporcional ya estaba escrito; lo tenía apagado
  // la telemetría que faltaba, no el diseño.
  //
  // Un nodo sin medida conserva 1 y se declara NO medido, en vez de pasar por
  // informado en silencio — el mismo criterio que `estimateLinkLatencyMs`.
  const decodeScales = deriveDecodeScales(profiledExecutors.map(({ executor, worker }) => ({
    nodeId: executor.nodeId,
    measuredTokensPerSecond: measuredDecodeThroughput(worker, baseConfig.model.publicName),
  })));
  const decodeScaleById = new Map(
    decodeScales.scales.map((scale) => [scale.nodeId, scale]),
  );

  const nodes = profiledExecutors.map(({ worker, executor, performance }) => {
    const memoryMiB = worker.capabilities.gpus.reduce(
      (sum, gpu) => sum + gpu.offeredVramMb,
      0,
    );
    const measuredPower = worker.capabilities.gpus.reduce(
      (sum, gpu) => sum + (gpu.powerW ?? 0),
      0,
    );
    return {
      id: executor.nodeId,
      region: worker.capabilities.region,
      endpoint: { host: executor.stageHost, port: executor.stagePort },
      memoryMiB,
      reserveMiB: Math.min(256, Math.max(0, memoryMiB - 1)),
      decodeScale: decodeScaleById.get(executor.nodeId)?.measured
        ? decodeScaleById.get(executor.nodeId)!.decodeScale
        : performance.decodeScale,
      prefillScale: performance.prefillScale,
      codecScale: performance.codecScale,
      powerWatts: measuredPower > 0 ? measuredPower : 1,
      availability: Math.max(0.01, Math.min(1, worker.reliability)),
      agent: { kind: "managed" as const },
    };
  });
  const observationByLink = new Map(runtimeLinkObservations.map((observation) => [
    linkKey(observation.fromNodeId, observation.toNodeId),
    observation,
  ]));
  const measuredLinks = profiledExecutors.flatMap((from) => profiledExecutors
    .filter((to) => to.executor.nodeId !== from.executor.nodeId)
    .flatMap((to) => {
      const observation = observationByLink.get(
        linkKey(from.executor.nodeId, to.executor.nodeId),
      );
      if (!observation) return [];
      return [{
        from: from.executor.nodeId,
        to: to.executor.nodeId,
        oneWayLatencyMs: Math.max(0.05, observation.rttP50Ms / 2),
        jitterP95Ms: Math.max(0, (observation.rttP95Ms - observation.rttP50Ms) / 2),
        bandwidthMbps: observation.goodputMbpsP50,
        // Probe failures are modeled as reachability below. Treating them as
        // both loss and unavailability would charge the same failure twice.
        lossRate: 0,
        availability: observation.availability,
        evidence: {
          source: "runtime-probe" as const,
          measuredAt: observation.measuredAt,
          validUntil: observation.measuredAt + RUNTIME_LINK_EVIDENCE_TTL_MS,
          successfulSamples: observation.successfulSamples,
          failedSamples: observation.failedSamples,
        },
      }];
    }));
  const reciprocalNodeIds = new Set(measuredLinks.flatMap((link) => (
    measuredLinks.some((candidate) =>
      candidate.from === link.to && candidate.to === link.from
    )
      ? [link.from, link.to]
      : []
  )));
  const routableNodes = nodes.filter((node) => reciprocalNodeIds.has(node.id));
  if (routableNodes.length < 2) {
    return {
      capacityNodes,
      config: null,
      readinessDetails: executors.map(({ executor }) =>
        reciprocalNodeIds.has(executor.nodeId)
          ? `${executor.nodeId}: runtime and reciprocal network link verified.`
          : `${executor.nodeId}: runtime verified; waiting for a reciprocal network link measurement.`
      ),
    };
  }
  const links = measuredLinks.filter(
    (link) => reciprocalNodeIds.has(link.from) && reciprocalNodeIds.has(link.to),
  );
  const rootHost = routableNodes[0]!.endpoint.host;
  const config = parseAutoDistributionConfig({
    ...structuredClone(baseConfig),
    nodes: routableNodes,
    links,
    distribution: {
      ...baseConfig.distribution,
      minimumStages: Math.min(baseConfig.distribution.minimumStages, routableNodes.length),
      maximumStages: Math.min(baseConfig.distribution.maximumStages, routableNodes.length),
    },
    runtime: {
      ...baseConfig.runtime,
      // Remote desktop executors resolve this through their packaged runtime
      // PATH. Never send the coordinator's absolute Linux interpreter path.
      stagePythonExecutable: "python",
      apiAdvertiseHost: rootHost,
      returnEndpoint: { ...baseConfig.runtime.returnEndpoint, host: rootHost },
    },
  });
  return { capacityNodes, config };
}

/** True only for a currently published model route that runtime telemetry marks as degraded. */
export function modelHasGpuFallback(
  modelId: string,
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
): boolean {
  return workers
    .filter((worker) => connectedWorkerIds.has(worker.id) && worker.status === "online")
    .flatMap((worker) => worker.capabilities.deployments)
    .filter((deployment) => deployment.model === modelId)
    .some((deployment) => {
      const execution = deployment.execution;
      if (!execution) return false;
      return execution.fallback || execution.stages?.some((stage) => stage.fallback) === true;
    });
}

/**
 * A fallback route is recycled only after enough physical executors have
 * re-published verified GPU capacity for the whole requested topology. CPU
 * availability keeps inference alive while this condition is false.
 */
export function verifiedGpuCapacityCanRepairModel(
  model: StoredRequestedModel,
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
): boolean {
  const minimumStageVramMiB = positiveProfileNumber(model.profile, "minimumStageVramMiB") ?? 512;
  const requiredVramMiB = positiveProfileNumber(model.profile, "requiredVramMiB")
    ?? minimumStageVramMiB * model.minimumNodes;
  const candidates = connectedExecutors(workers, connectedWorkerIds)
    .filter(({ executor }) => executor.computeMode !== "cpu-only")
    .map(({ worker }) => ({
      availableVramMiB: worker.capabilities.gpus
        .filter((gpu) => gpu.vendor.trim().toLowerCase() !== "cpu" && gpu.physicalVramMb > 0)
        .reduce((sum, gpu) => sum + gpu.freeOfferedVramMb, 0),
    }))
    .filter((candidate) => candidate.availableVramMiB >= minimumStageVramMiB);
  return candidates.length >= model.minimumNodes
    && candidates.reduce((sum, candidate) => sum + candidate.availableVramMiB, 0) >= requiredVramMiB;
}

function positiveProfileNumber(profile: Record<string, unknown> | null, key: string): number | null {
  const value = profile?.[key];
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

export function resolveConnectedExecutorAgent(
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
  hub: WorkerHub,
  nodeId: string,
  launch: PythonPipelineLaunchDescription,
): LaunchAgent | undefined {
  const entry = connectedExecutors(workers, connectedWorkerIds)
    .find(({ executor }) => executor.nodeId === nodeId);
  return entry
    ? new WorkerTunnelLaunchAgent(hub, entry.worker.id, nodeId, launch)
    : undefined;
}

function connectedExecutors(
  workers: readonly StoredWorker[],
  connectedWorkerIds: ReadonlySet<string>,
): ConnectedExecutor[] {
  return workers
    .filter((worker) => connectedWorkerIds.has(worker.id) && worker.capabilities.distributedExecutor)
    .filter((worker) => executorCapacityIsEligible(worker))
    .map((worker) => ({ worker, executor: worker.capabilities.distributedExecutor! }))
    .filter(({ executor }) => executor.protocol === "gdlp-worker-tunnel/2")
    .filter((entry, index, all) => all.findIndex(
      (candidate) => candidate.executor.nodeId === entry.executor.nodeId,
    ) === index);
}

function executorCapacityIsEligible(worker: StoredWorker): boolean {
  const executor = worker.capabilities.distributedExecutor;
  if (!executor) return false;
  const hasVerifiedGpu = worker.capabilities.gpus.some(
    (gpu) => gpu.vendor.trim().toLowerCase() !== "cpu",
  );
  const mode = executor.computeMode ?? "automatic";
  if (mode === "gpu-only") return hasVerifiedGpu;
  if (mode === "cpu-only") return executor.cpuEligible === true;
  // Automatic mode can use a verified GPU immediately. CPU is accepted only
  // after a current desktop client explicitly announces that fallback is
  // allowed; old clients remain GPU-only by default.
  return hasVerifiedGpu || executor.cpuEligible === true;
}

function linkKey(fromNodeId: string, toNodeId: string): string {
  return `${fromNodeId}\u0000${toNodeId}`;
}

function eligiblePlannerPerformance(
  entry: ConnectedExecutor,
): PlannerPerformanceScales | null {
  const evidence = entry.executor.performanceEvidence;
  if (!evidence) return null;
  const profile = evidence.profile;
  try {
    const scales = plannerScalesFromCoordinatorEvidence(evidence, {
      workerId: entry.worker.id,
      nodeId: entry.executor.nodeId,
    });
    const mode = entry.executor.computeMode ?? "automatic";
    if (profile.backend === "cpu") {
      if (entry.executor.cpuEligible !== true || mode === "gpu-only") return null;
      return scales;
    }
    if (mode === "cpu-only") return null;
    const acceleration = entry.executor.acceleration;
    if (
      acceleration?.state !== "gpu-ready"
      || acceleration.backend !== profile.backend
      || !acceleration.deviceName
      || normalizeDeviceName(acceleration.deviceName)
        !== normalizeDeviceName(profile.deviceName)
    ) {
      return null;
    }
    const expectedVendor = {
      cuda: "nvidia",
      rocm: "amd",
      mps: "apple",
      xpu: "intel",
    }[profile.backend];
    if (!entry.worker.capabilities.gpus.some(
      (gpu) => gpu.vendor.trim().toLowerCase() === expectedVendor,
    )) {
      return null;
    }
    return scales;
  } catch {
    // Capacity remains visible, but stale, noisy, tampered, mismatched or
    // otherwise unverifiable performance evidence cannot enter a route.
    return null;
  }
}

function normalizeDeviceName(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}
