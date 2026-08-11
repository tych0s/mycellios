import { dirname, join } from "node:path";
import { readFileSync } from "node:fs";
import { loadWorkerConfig } from "../core/config.js";
import { WorkerAgent } from "../worker/agent.js";
import {
  createHeadlessRuntime,
  headlessEnvironmentFromNodeConfiguration,
} from "../worker/headless-runtime.js";
import { NodeConfigurationStore } from "./config-store.js";
import { MycelliosNodeServiceHost } from "./service-host.js";
import { buildNodeDiagnosticSnapshot, writeNodeDiagnosticSnapshot } from "./diagnostics.js";
import {
  LinuxSystemdCredentialStore,
  MacOsKeychainSecretStore,
  NodeIdentityStore,
  WindowsDpapiSecretStore,
  type ProtectedSecretStore,
} from "./identity-store.js";
import { breachedLimit, createNodeResourceObserver, NodeResourceGovernor } from "./resource-governor.js";
import { assertNodeOsIsolation } from "./os-isolation.js";
import { NodeControlStateStore } from "./control-state.js";
import { NodeCommandExecutor } from "./command-executor.js";
import { runNodeCommandClient } from "./command-client.js";
import { NodeCommandApplicator } from "./command-applicator.js";
import { NodeComponentLifecycle, resolveNodePythonProduct } from "./component-lifecycle.js";
import { acquireComponentInUseLease, type ComponentInUseLease } from "../update/component-update-manager.js";
import { resolveNodeSourceRevision } from "./build-identity.js";
import { NodeReconciliationStateStore } from "./reconciliation-state.js";
import { MYCELLIOS_RUNTIME_ABI } from "../update/runtime-compatibility.js";
import { NodeUninstallScheduler } from "./uninstall-helper.js";
import { NodeEnrollmentBootstrap } from "./enrollment-bootstrap.js";
import { DEFAULT_NODE_WORK_POLICY, evaluateNodeWorkPolicy } from "../contracts/node-work-policy.js";

const configPath = argument("--config") ?? defaultNodeConfigurationPath();
const allowLegacyMigration = process.argv.includes("--confirm-legacy-migration");
const store = new NodeConfigurationStore(configPath, { allowLegacyMigration });
const loaded = await store.load();
const config = loaded.config;
if (loaded.migrated) await store.save(config);
const version = packageVersion();
const sourceRevision = await resolveNodeSourceRevision();
const environment = headlessEnvironmentFromNodeConfiguration(config, version);
const componentLifecycle = config.componentUpdates
  ? new NodeComponentLifecycle(config, version, async () => true, async () => async () => undefined)
  : null;
let pendingComponentActivation = componentLifecycle
  ? await componentLifecycle.pendingActivationIsCurrent()
  : false;
if (componentLifecycle && !pendingComponentActivation && await componentLifecycle.pendingActivation()) {
  await componentLifecycle.markActivationHealthy();
}
const host = new MycelliosNodeServiceHost(join(dirname(configPath), "state"));
const controlStore = new NodeControlStateStore(join(host.stateDirectory, "control.json"));
await host.acquire(config.revision);
await assertNodeOsIsolation(config);
const controlState = await controlStore.load();

let agent: WorkerAgent | null = null;
let governor: NodeResourceGovernor | null = null;
let commandClientAbort: AbortController | null = null;
let commandClient: Promise<void> | null = null;
let failure: unknown = null;
let lastIncident: { code: string; occurredAt: string } | null = null;
let componentInUseLease: ComponentInUseLease | null = null;
try {
  const activePythonProduct = await resolveNodePythonProduct(config);
  if (activePythonProduct && componentLifecycle) {
    componentInUseLease = await acquireComponentInUseLease(componentLifecycle.storageRoot, ["python-product"]);
    environment.pythonPath = componentInUseLease.roots["python-product"]!;
  }
  const runtime = await createHeadlessRuntime(environment);
  const device = runtime.physicalProbe.devices[0];
  if (!device) throw new Error("mycellios_node_physical_probe_missing_device");
  const detectedVramMiB = Math.floor(
    Math.min(device.totalMemoryBytes, device.runtimeTotalMemoryBytes) / (1024 * 1024),
  );
  const workerConfig = loadWorkerConfig(environment.configPath);
  const admissionSigner = await new NodeIdentityStore(
    config.coordinator.identityPath,
    platformSecretStore(),
    { allowLegacyMigration },
  ).loadOrCreate(config.nodeId);
  const enrollmentBootstrap = new NodeEnrollmentBootstrap(
    argument("--enrollment-file") ?? join(dirname(configPath), "enrollment.json"),
    config.coordinator.url,
  );
  agent = new WorkerAgent({
    ...workerConfig,
    offeredVramMb: Math.min(detectedVramMiB, config.limits.maxVramMiB || detectedVramMiB),
    limits: {
      ...workerConfig.limits,
      maxConcurrency: Math.min(workerConfig.limits.maxConcurrency, config.limits.maxConcurrency),
    },
  }, {
    coordinatorUrl: environment.coordinatorUrl,
    identity: { kind: "device", id: config.nodeId },
    admissionSigner,
    beforeSignedAdmission: (input) => enrollmentBootstrap.redeem(input),
    reconnect: true,
    advertiseDeployment: false,
    agentVersion: version,
    verifiedGpuRuntime: runtime.verifiedGpuRuntime,
    preferredHardwareGpu: runtime.preferredHardwareGpu,
    hardwareCapacityOverride: {
      id: `physical-${device.index}`,
      vendor: runtime.preferredHardwareGpu.vendor,
      model: device.name,
      physicalVramMb: detectedVramMiB,
    },
    distributedExecutor: runtime.executor,
    contributionControl: {
      initialEnabled: controlState.contributionEnabled && !controlState.draining,
      onRemoteChange: async (enabled) => { await controlStore.setContributionEnabled(enabled); },
    },
    workAdmissionPolicy: (model, at) => evaluateNodeWorkPolicy(config.policy ?? DEFAULT_NODE_WORK_POLICY, model, at),
  });
  const resourceObserver = createNodeResourceObserver(config.runtime.cachePath);
  const snapshotResourceObserver = createNodeResourceObserver(config.runtime.cachePath);
  governor = new NodeResourceGovernor(
    config.limits,
    resourceObserver,
    async (code) => {
      lastIncident = { code, occurredAt: new Date().toISOString() };
      await host.writeHealth("failed", config.revision, code);
      await controlStore.setContributionEnabled(false);
      await agent?.setContributionEnabled(false).catch(() => undefined);
      await agent?.stop();
    },
  );
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.once(signal, () => void agent?.stop());
  }
  const running = agent.start();
  await waitUntil(() => agent?.isReady === true, running, 60_000);
  const uninstall = config.uninstall
    ? new NodeUninstallScheduler(config.uninstall.manifestPath, join(host.stateDirectory, "uninstall-requests"))
    : undefined;
  const commandApplicator = new NodeCommandApplicator(agent, controlStore, store, config, componentLifecycle ?? undefined, uninstall);
  if (controlState.draining) await commandApplicator.restoreDrain();
  await host.writeHealth("ready", config.revision);
  await writeNodeDiagnosticSnapshot(
    join(host.stateDirectory, "diagnostics.json"),
    buildNodeDiagnosticSnapshot({
      config,
      version,
      state: "ready",
      physicalProbe: runtime.physicalProbe,
    }),
  );
  governor.start();
  if (pendingComponentActivation && componentLifecycle) {
    await componentLifecycle.markActivationHealthy();
    pendingComponentActivation = false;
    await controlStore.setDraining(false);
    if (controlState.contributionEnabled) await agent.setContributionEnabled(true);
  }
  const controlSession = agent.nodeControlSession;
  if (controlSession) {
    commandClientAbort = new AbortController();
    const commandExecutor = new NodeCommandExecutor(
      join(host.stateDirectory, `commands-generation-${controlSession.generation}.json`),
      config.nodeId,
      controlSession.generation,
    );
    const reconciliationState = new NodeReconciliationStateStore(
      join(host.stateDirectory, "reconciliation.json"),
      config.nodeId,
    );
    let restartAfterReconciliation = false;
    commandClient = runNodeCommandClient({
      coordinatorUrl: environment.coordinatorUrl,
      nodeId: config.nodeId,
      session: () => agent?.nodeControlSession ?? null,
      executor: commandExecutor,
      apply: async (application) => commandApplicator.apply(application),
      signal: commandClientAbort.signal,
      reconciliation: {
        state: reconciliationState,
        snapshot: async (cursor, session) => {
          const [currentControl, currentConfig, journal, resourceObservation] = await Promise.all([
            controlStore.load(),
            store.load(),
            commandExecutor.inspect(),
            snapshotResourceObserver().catch(() => null),
          ]);
          const resourceViolation = resourceObservation ? breachedLimit(currentConfig.config.limits, resourceObservation) : "node_resource_probe_failed";
          const incident = lastIncident ?? (resourceViolation ? { code: resourceViolation, occurredAt: new Date().toISOString() } : null);
          return {
            schema: "mycellios-node-snapshot/1",
            nodeId: config.nodeId,
            generation: session.generation,
            cursor,
            observedAt: new Date().toISOString(),
            state: incident ? "degraded" : agent?.isReady ? "ready" : "reconnecting",
            contributionEnabled: currentControl.contributionEnabled,
            draining: currentControl.draining,
            updateChannel: currentConfig.config.updateChannel,
            limits: currentConfig.config.limits,
            policy: currentConfig.config.policy ?? DEFAULT_NODE_WORK_POLICY,
            activeCommandIds: journal.records.filter(({ state }) => state === "applying").map(({ commandId }) => commandId),
            build: { version, sourceRevision },
            runtime: { ready: agent?.isReady === true, abi: MYCELLIOS_RUNTIME_ABI, backend: runtime.verifiedGpuRuntime.backend,
              uninstallAvailable: config.uninstall !== undefined },
            diagnostics: {
              capturedAt: new Date().toISOString(),
              configRevision: currentConfig.config.revision,
              capacity: {
                acceleratorCount: runtime.physicalProbe.devices.length,
                primaryAccelerator: device ? { name: device.name, totalMemoryMiB: detectedVramMiB,
                  offeredMemoryMiB: Math.min(detectedVramMiB, currentConfig.config.limits.maxVramMiB || detectedVramMiB) } : null,
              },
              resources: resourceObservation ? { ...resourceObservation, healthy: resourceViolation === null, violation: resourceViolation } : null,
              incident,
            },
          };
        },
        onReconciled: async ({ desiredState, acknowledgedCommandIds }) => {
          if (uninstall && await uninstall.armAcknowledged(new Set(acknowledgedCommandIds)) > 0) {
            await agent?.stop();
            return;
          }
          const currentControl = await controlStore.load();
          if (desiredState.drain !== currentControl.draining) await commandApplicator.reconcileDrain(desiredState.drain);
          if (currentControl.contributionEnabled !== desiredState.contributionEnabled) {
            await controlStore.setContributionEnabled(desiredState.contributionEnabled);
          }
          await agent?.setContributionEnabled(desiredState.contributionEnabled && !desiredState.drain);
          const current = await store.load();
          const limitsChanged = JSON.stringify(current.config.limits) !== JSON.stringify(desiredState.limits);
          const policyChanged = JSON.stringify(current.config.policy) !== JSON.stringify(desiredState.policy);
          if (limitsChanged || policyChanged || current.config.updateChannel !== desiredState.updateChannel) {
            await store.save({
              ...current.config,
              revision: current.config.revision + 1,
              limits: desiredState.limits,
              policy: desiredState.policy ?? DEFAULT_NODE_WORK_POLICY,
              updateChannel: desiredState.updateChannel,
            });
            restartAfterReconciliation = true;
          }
        },
        onSnapshotAcknowledged: async () => {
          lastIncident = null;
          if (restartAfterReconciliation) await agent?.stop();
        },
      },
      onError: (error) => {
        const code = safeIncidentCode(error instanceof Error ? error.message : "node_command_client_error");
        lastIncident = { code, occurredAt: new Date().toISOString() };
        void host.writeHealth("ready", config.revision, code).catch(() => undefined);
      },
      onAcknowledged: async (command, result) => {
        if (result.state === "applied" && (commandApplicator.takeRestartAfterAck(command.id) || await commandApplicator.armStopAfterAck(command.id))) {
          await agent?.stop();
        }
      },
    });
  }
  await running;
} catch (error) {
  failure = error;
  if (pendingComponentActivation && componentLifecycle) {
    try {
      await componentLifecycle.rollbackFailedActivation();
      pendingComponentActivation = false;
    } catch (rollbackError) {
      failure = new AggregateError([error, rollbackError], "node_component_activation_and_rollback_failed");
    }
  }
  throw failure;
} finally {
  commandClientAbort?.abort();
  await commandClient?.catch(() => undefined);
  governor?.stop();
  await agent?.stop().catch(() => undefined);
  await componentInUseLease?.release().catch(() => undefined);
  await host.release(
    config.revision,
    failure ? "failed" : "stopped",
    failure instanceof Error ? failure.message : failure ? String(failure) : undefined,
  );
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function defaultNodeConfigurationPath(): string {
  if (process.platform === "win32") {
    const root = process.env.PROGRAMDATA;
    if (!root) throw new Error("mycellios_node_programdata_is_missing");
    return join(root, "mycellios", "node.json");
  }
  if (process.platform === "darwin") return "/Library/Application Support/mycellios/node.json";
  return "/etc/mycellios/node.json";
}

async function waitUntil(
  predicate: () => boolean,
  running: Promise<void>,
  timeoutMs: number,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("mycellios_node_ready_timeout");
    await Promise.race([
      running.then(() => { throw new Error("mycellios_node_stopped_before_ready"); }),
      new Promise<void>((resolve) => setTimeout(resolve, 100)),
    ]);
  }
}

function packageVersion(): string {
  const document = JSON.parse(
    readFileSync(new URL("../../package.json", import.meta.url), "utf8"),
  ) as { version?: unknown };
  if (typeof document.version !== "string" || !document.version.trim()) {
    throw new Error("mycellios_node_package_version_is_invalid");
  }
  return document.version;
}

function platformSecretStore(): ProtectedSecretStore {
  if (process.platform === "linux") return new LinuxSystemdCredentialStore(
    join(dirname(config.coordinator.identityPath), "protected"),
  );
  if (process.platform === "darwin") return new MacOsKeychainSecretStore();
  if (process.platform === "win32") {
    const root = process.env.PROGRAMDATA;
    if (!root) throw new Error("mycellios_node_programdata_is_missing");
    return new WindowsDpapiSecretStore(join(root, "mycellios", "protected-identity"));
  }
  throw new Error(`mycellios_node_protected_identity_backend_is_unsupported:${process.platform}`);
}

function safeIncidentCode(value: string): string {
  const code = value.match(/^[a-zA-Z0-9_.:-]{1,120}/)?.[0];
  if (!code || /(authorization|cookie|credential|password|secret|token|api[-_]?key)/i.test(code) || /[A-Za-z0-9_-]{32,}/.test(code)) {
    return "unclassified_error";
  }
  return code;
}
