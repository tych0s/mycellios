import type { AutoDistributionConfig } from "../distribution/auto-distribute.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";
import type { LaunchAgent } from "../distribution/launch-supervisor.js";
import type { PythonPipelineLaunchDescription } from "../distribution/python-launcher.js";
import { WorkerTunnelLaunchAgent } from "../distribution/worker-tunnel-launch-agent.js";
import type { StoredWorker } from "../storage/store.js";
import type { WorkerHub } from "./worker-hub.js";
import type { DynamicActivationSnapshot } from "./model-activation-manager.js";

interface ConnectedExecutor {
  worker: StoredWorker;
  executor: NonNullable<StoredWorker["capabilities"]["distributedExecutor"]>;
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
      decodeScale: 1,
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
      oneWayLatencyMs: Math.max(
        0.1,
        (from.worker.capabilities.network.coordinatorRttMs
          + to.worker.capabilities.network.coordinatorRttMs) / 2,
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
      availability: Math.max(
        0.01,
        Math.min(from.worker.reliability, to.worker.reliability),
      ),
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
      apiAdvertiseHost: rootHost,
      returnEndpoint: { ...baseConfig.runtime.returnEndpoint, host: rootHost },
    },
  });
  return { capacityNodes, config };
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
    .map((worker) => ({ worker, executor: worker.capabilities.distributedExecutor! }))
    .filter((entry, index, all) => all.findIndex(
      (candidate) => candidate.executor.nodeId === entry.executor.nodeId,
    ) === index);
}
