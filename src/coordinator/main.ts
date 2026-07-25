import { readFile } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import { loadCoordinatorConfig } from "../core/config.js";
import { parseAutoDistributionConfig } from "../distribution/auto-distribute.js";
import {
  buildConnectedExecutorActivationSnapshot,
  resolveConnectedExecutorAgent,
} from "./connected-executor-activation.js";
import {
  AutomaticModelActivationManager,
  DynamicModelActivationManager,
} from "./model-activation-manager.js";
import { createCoordinator } from "./server.js";

const config = loadCoordinatorConfig();
const activationConfigPath = process.env.MYCELLIOS_AUTO_DISTRIBUTE_CONFIG?.trim();
const dynamicWorkerActivation = process.env.MYCELLIOS_DYNAMIC_WORKER_ACTIVATION?.trim() === "1";
const absoluteActivationConfigPath = activationConfigPath
  ? isAbsolute(activationConfigPath) ? activationConfigPath : resolve(process.cwd(), activationConfigPath)
  : undefined;
const baseActivationConfig = absoluteActivationConfigPath
  ? parseAutoDistributionConfig(JSON.parse(await readFile(absoluteActivationConfigPath, "utf8")) as unknown)
  : undefined;
const activationManager = baseActivationConfig && !dynamicWorkerActivation
  ? new AutomaticModelActivationManager(baseActivationConfig)
  : undefined;
const runtime = await createCoordinator(config, {
  logger: true,
  ...(activationManager ? { activationManager } : {}),
  ...(baseActivationConfig && dynamicWorkerActivation
    ? {
        activationManagerFactory: ({ store, hub, deploymentController }) => new DynamicModelActivationManager({
          snapshot: () => buildConnectedExecutorActivationSnapshot(
            baseActivationConfig,
            store.listWorkers(),
            hub.connectedWorkerIds(),
            hub.runtimeLinkObservations(),
          ),
          resolveManagedAgent: (nodeId, launch) => resolveConnectedExecutorAgent(
            store.listWorkers(),
            hub.connectedWorkerIds(),
            hub,
            nodeId,
            launch,
          ),
          loadProgress: (modelId) => store.listActivationEvents(modelId),
          onProgress: (modelId, event) => {
            store.appendActivationEvent(modelId, event);
            if (event.phase === "running_canary") {
              const operation = deploymentController.activeOperationForModel(modelId);
              if (operation) deploymentController.markCanary(operation.id);
            }
          },
          onPlanPrepared: (modelId, stages) => {
            const operation = deploymentController.activeOperationForModel(modelId);
            if (!operation) throw new Error(`deployment_operation_missing:${modelId}`);
            return deploymentController.prepareRoute(operation.id, stages).id;
          },
          onActivated: (modelId, reservationId, result) => {
            const canary = {
              passed: true,
              text: result.canaryText,
              ...result.canaryMetrics,
              workerId: result.workerId,
            };
            if (reservationId) {
              deploymentController.commitRoute(reservationId, canary);
              return;
            }
            const operation = deploymentController.activeOperationForModel(modelId);
            if (operation) deploymentController.completeOperation(operation.id, "active", { canary });
          },
        }),
      }
    : {}),
});

await runtime.app.listen({ host: config.host, port: config.port });
runtime.app.log.info(
  `mycellios coordinator listening on http://${config.host}:${config.port}`,
);
if (baseActivationConfig) {
  runtime.app.log.info(
    `${dynamicWorkerActivation ? "dynamic worker" : "static"} model activation enabled with ${activationConfigPath}`,
  );
}

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.once(signal, () => {
    void runtime.close().finally(() => process.exit(0));
  });
}
