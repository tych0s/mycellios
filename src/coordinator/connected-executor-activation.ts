import type { AutoDistributionConfig } from "../distribution/auto-distribute.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";
import type { LaunchAgent } from "../distribution/launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "../distribution/python-launcher.js";
import { WorkerTunnelLaunchAgent } from "../distribution/worker-tunnel-launch-agent.js";
import type { StoredRequestedModel, StoredWorker } from "../storage/store.js";
import type { WorkerHub } from "./worker-hub.js";
import type { DynamicActivationSnapshot } from "./model-activation-manager.js";
import type { RuntimeLinkObservation } from "./runtime-link-observations.js";
import {
  plannerScalesFromProfile,
  type PlannerPerformanceScales,
} from "../performance/runtime-profile.js";

interface ConnectedExecutor {
  worker: StoredWorker;
  executor: NonNullable<StoredWorker["capabilities"]["distributedExecutor"]>;
}

interface ProfiledConnectedExecutor extends ConnectedExecutor {
  performance: PlannerPerformanceScales;
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
  if (profiledExecutors.length < 2) return { capacityNodes, config: null };

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
      decodeScale: performance.decodeScale,
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
  if (routableNodes.length < 2) return { capacityNodes, config: null };
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
  const profile = entry.executor.performanceProfile;
  if (!profile) return null;
  try {
    const scales = plannerScalesFromProfile(profile);
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
