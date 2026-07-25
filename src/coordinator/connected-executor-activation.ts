import type { AutoDistributionConfig } from "../distribution/auto-distribute.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";
import type { LaunchAgent } from "../distribution/launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "../distribution/python-launcher.js";
import { WorkerTunnelLaunchAgent } from "../distribution/worker-tunnel-launch-agent.js";
import type { StoredRequestedModel, StoredWorker } from "../storage/store.js";
import { deriveDecodeScales } from "../distribution/node-scale.js";
import type { WorkerHub } from "./worker-hub.js";
import type { DynamicActivationSnapshot } from "./model-activation-manager.js";

interface ConnectedExecutor {
  worker: StoredWorker;
  executor: NonNullable<StoredWorker["capabilities"]["distributedExecutor"]>;
}

/**
 * Decode throughput this worker actually measured, or null.
 *
 * Only `measured` counts. An estimated or configured number is a guess about
 * hardware, and planning a layer split on a guess is how the planner ended up
 * trusting a constant in the first place.
 */
function measuredDecodeThroughput(worker: StoredWorker): number | null {
  const measured = worker.capabilities.deployments
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
): DynamicActivationSnapshot {
  const executors = connectedExecutors(workers, connectedWorkerIds);
  const capacityNodes = executors.map(({ worker, executor }) => ({
    id: executor.nodeId,
    availableVramMiB: worker.capabilities.gpus.reduce(
      (sum, gpu) => sum + gpu.freeOfferedVramMb,
      0,
    ),
  }));
  if (executors.length < 2) return { capacityNodes, config: null };

  // El SEGUNDO cero del planificador. `coordinatorRttMs` ya se mide, pero
  // `decodeScale` seguía fijado a 1 aquí, así que `ProportionalComputePlanner`
  // dividía por un vector de unos y **el reparto proporcional degeneraba a
  // reparto igual**: el planificador no podía distinguir una 4090 de una
  // 1050 Ti. El planificador proporcional ya estaba escrito; lo tenía apagado
  // la telemetría que faltaba, no el diseño.
  //
  // Un nodo sin medida conserva 1 y se declara NO medido, en vez de pasar por
  // informado en silencio — el mismo criterio que `estimateLinkLatencyMs`.
  const decodeScales = deriveDecodeScales(executors.map(({ executor, worker }) => ({
    nodeId: executor.nodeId,
    measuredTokensPerSecond: measuredDecodeThroughput(worker),
  })));
  const decodeScaleById = new Map(
    decodeScales.scales.map((scale) => [scale.nodeId, scale.decodeScale]),
  );

  const nodes = executors.map(({ worker, executor }) => {
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
      decodeScale: decodeScaleById.get(executor.nodeId) ?? 1,
      prefillScale: 1,
      codecScale: 1,
      powerWatts: measuredPower > 0 ? measuredPower : 1,
      availability: Math.max(0.01, Math.min(1, worker.reliability)),
      agent: { kind: "managed" as const },
    };
  });
  const links = executors.flatMap((from) => executors
    .filter((to) => to.executor.nodeId !== from.executor.nodeId)
    .map((to) => ({
      from: from.executor.nodeId,
      to: to.executor.nodeId,
      oneWayLatencyMs: estimateLinkLatencyMs(
        from.worker.capabilities.network.coordinatorRttMs,
        to.worker.capabilities.network.coordinatorRttMs,
      ),
      jitterP95Ms: 0,
      bandwidthMbps: Math.max(
        1,
        Math.min(
          from.worker.capabilities.network.uplinkMbps,
          to.worker.capabilities.network.downlinkMbps,
        ),
      ),
      lossRate: 0,
      // Worker reliability is already represented on both endpoint nodes.
      // Reusing it here would count the same failures again for the forward
      // and return links, making two 95% reliable workers look <90% reliable.
      availability: 1,
    })));
  const rootHost = nodes[0]!.endpoint.host;
  const config = parseAutoDistributionConfig({
    ...structuredClone(baseConfig),
    nodes,
    links,
    distribution: {
      ...baseConfig.distribution,
      minimumStages: Math.min(baseConfig.distribution.minimumStages, nodes.length),
      maximumStages: Math.min(baseConfig.distribution.maximumStages, nodes.length),
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
