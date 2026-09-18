import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { randomBytes } from "node:crypto";
import { createServer } from "node:net";
import { homedir, tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import process from "node:process";
import { workerConfigSchema, type WorkerConfig } from "../contracts/schemas.js";
import { readNativeRuntimeBuildMetadata } from "../core/native-build-identity.js";
import {
  parseAutoDistributionConfig,
  type AutoDistributionConfig,
  type AutoDistributionRunResult,
} from "../distribution/auto-distribute.js";
import {
  LocalProcessAgent,
  type LaunchAgent,
  type LaunchAgentStartRequest,
  type LaunchProcessHandle,
  type RuntimePreparationProgressEvent,
} from "../distribution/launch-supervisor.js";
import type {
  PythonLaunchProcess,
  PythonPipelineLaunchDescription,
} from "../distribution/python-launcher.js";
import {
  resolveConnectedExecutorAgent,
} from "../coordinator/connected-executor-activation.js";
import { DynamicModelActivationManager } from "../coordinator/model-activation-manager.js";
import type { HubModelCapacityProfile } from "../coordinator/model-catalog.js";
import { createCoordinator, type CoordinatorRuntime } from "../coordinator/server.js";
import { NodeEnrollmentStore } from "../coordinator/node-enrollment-store.js";
import { workerAdmissionPublicKeyFingerprint } from "../coordinator/worker-admission.js";
import {
  MODEL_ADAPTER_EVIDENCE_SCOPE,
  MODEL_ADAPTER_REGISTRY_ID,
  resolveModelAdapterContract,
} from "../contracts/model-adapter-registry.js";
import { WorkerAgent } from "../worker/agent.js";
import {
  generateWorkerAdmissionCredential,
  workerAdmissionSigner,
} from "../worker/admission-credential.js";
import { prepareNodeStageArtifacts } from "../worker/stage-artifact-preparer.js";

const DEFAULT_MODEL = "hmellor/tiny-random-LlamaForCausalLM";
const DEFAULT_TIMEOUT_SECONDS = 300;
const CLEANUP_STEP_TIMEOUT_MS = 30_000;

interface DevelopmentGateCliOptions {
  help: boolean;
  modelSource: string;
  runtimeRoot?: string;
  timeoutMs: number;
}

export interface DevelopmentGateCleanupStep {
  name: string;
  run(): void | Promise<void>;
}

export interface DevelopmentSignalSource {
  once(event: "SIGINT" | "SIGTERM", listener: () => void): unknown;
  removeListener(event: "SIGINT" | "SIGTERM", listener: () => void): unknown;
}

export interface DevelopmentSignalBinding {
  signal: Promise<"SIGINT" | "SIGTERM">;
  dispose(): void;
}

export interface DevelopmentGateEvent {
  sequence: number;
  at: string;
  phase: string;
  message: string;
}

export interface DevelopmentGateResult {
  evidenceClass: "cpu-loopback-logical";
  model: string;
  stages: number;
  signedWorkers: number;
  directRuntimeLinks: number;
  completionTokens: number;
  canaryPassed: true;
  routePassed: true;
}

export interface DevelopmentGateExecutionOptions {
  workspace?: string | undefined;
  signal?: AbortSignal | undefined;
  installSignalHandlers?: boolean | undefined;
  writeOutput?: boolean | undefined;
  onEvent?: ((event: DevelopmentGateEvent) => void) | undefined;
}

/**
 * Runs all cleanup steps at most once. A failed step never prevents later
 * steps from draining workers, processes, the coordinator, or temporary data.
 */
export function createIdempotentDevelopmentCleanup(
  steps: readonly DevelopmentGateCleanupStep[],
  stepTimeoutMs = CLEANUP_STEP_TIMEOUT_MS,
): () => Promise<void> {
  if (!Number.isSafeInteger(stepTimeoutMs) || stepTimeoutMs < 1) {
    throw new Error("development_cleanup_timeout_invalid");
  }
  let running: Promise<void> | null = null;
  return () => {
    running ??= (async () => {
      const errors: Error[] = [];
      for (const step of steps) {
        try {
          await withTimeout(
            Promise.resolve().then(() => step.run()),
            stepTimeoutMs,
            `development_cleanup_timeout:${step.name}`,
          );
        } catch (error) {
          errors.push(normalizeError(error, `development_cleanup_failed:${step.name}`));
        }
      }
      if (errors.length > 0) {
        throw new AggregateError(errors, "development_gate_cleanup_failed");
      }
    })();
    return running;
  };
}

/**
 * Resolves on the first termination signal and removes both handlers
 * immediately. A second Ctrl+C therefore retains the operating system's normal
 * hard-stop behaviour if graceful cleanup itself becomes stuck.
 */
export function installDevelopmentSignalHandlers(
  source: DevelopmentSignalSource,
): DevelopmentSignalBinding {
  let settled = false;
  let resolveSignal!: (signal: "SIGINT" | "SIGTERM") => void;
  const signal = new Promise<"SIGINT" | "SIGTERM">((resolvePromise) => {
    resolveSignal = resolvePromise;
  });
  const onSigint = () => settle("SIGINT");
  const onSigterm = () => settle("SIGTERM");
  const dispose = () => {
    source.removeListener("SIGINT", onSigint);
    source.removeListener("SIGTERM", onSigterm);
  };
  const settle = (value: "SIGINT" | "SIGTERM") => {
    if (settled) return;
    settled = true;
    dispose();
    resolveSignal(value);
  };
  source.once("SIGINT", onSigint);
  source.once("SIGTERM", onSigterm);
  return { signal, dispose };
}

export async function executeDistributedDevelopmentGate(
  argv: readonly string[],
  environment: NodeJS.ProcessEnv = process.env,
  options: DevelopmentGateExecutionOptions = {},
): Promise<DevelopmentGateResult | null> {
  const cli = parseCliOptions(argv);
  if (cli.help) {
    if (options.writeOutput !== false) {
      process.stdout.write(`${helpText()}\n`);
    }
    return null;
  }

  const workspace = resolve(
    options.workspace ?? resolve(import.meta.dirname, "../.."),
  );
  const pythonSource = join(workspace, "python");
  const cacheRoot = join(workspace, "runtime", "dev-distributed-e2e-cache");
  const runtimeRoot = resolve(
    cli.runtimeRoot
      ?? environment.MYCELLIOS_DESKTOP_RUNTIME_ROOT
      ?? defaultRuntimeRoot(workspace, environment),
  );
  const pythonExecutable = process.platform === "win32"
    ? join(runtimeRoot, "python.exe")
    : join(runtimeRoot, "bin", "python");
  if (!existsSync(pythonExecutable)) {
    throw new Error(`Installed Mycellios Python is missing: ${pythonExecutable}`);
  }

  const controller = new AbortController();
  const forwardExternalAbort = () => {
    controller.abort(
      options.signal?.reason instanceof Error
        ? options.signal.reason
        : new Error("development_gate_cancelled"),
    );
  };
  if (options.signal?.aborted) forwardExternalAbort();
  else options.signal?.addEventListener("abort", forwardExternalAbort, {
    once: true,
  });
  const gateTimeout = setTimeout(
    () => controller.abort(new Error(`development_gate_timeout:${cli.timeoutMs}`)),
    cli.timeoutMs,
  );
  gateTimeout.unref();

  const temporaryRoot = await mkdtemp(join(tmpdir(), "mycellios-dev-e2e-"));
  const runtimeEnvironment: NodeJS.ProcessEnv = {
    ...environment,
    PYTHONPATH: pythonSource,
    HF_HOME: resolve(workspace, environment.MYCELLIOS_DEV_E2E_HF_HOME?.trim() || join(cacheRoot, "huggingface")),
    ...(cli.modelSource === DEFAULT_MODEL ? { HF_HUB_OFFLINE: "1" } : {}),
    TOKENIZERS_PARALLELISM: "false",
    PATH: [runtimeRoot, dirname(pythonExecutable), environment.PATH ?? ""]
      .filter(Boolean)
      .join(delimiter),
  };
  await Promise.all([
    mkdir(runtimeEnvironment.HF_HOME!, { recursive: true }),
    mkdir(join(cacheRoot, "node-a"), { recursive: true }),
    mkdir(join(cacheRoot, "node-b"), { recursive: true }),
  ]);

  let coordinator: CoordinatorRuntime | null = null;
  let activationManager: DynamicModelActivationManager | null = null;
  const workers: WorkerAgent[] = [];
  const workerRuns: Promise<void>[] = [];
  const launchAgents: DevelopmentStageLaunchAgent[] = [];
  let activatedResult: AutoDistributionRunResult | null = null;
  let signalBinding: DevelopmentSignalBinding | null = null;
  let interruptedSignal: "SIGINT" | "SIGTERM" | null = null;
  let eventSequence = 0;
  let diagnosticTimer: NodeJS.Timeout | undefined;
  const emitPhase = (phase: string, message: string): void => {
    const event: DevelopmentGateEvent = {
      sequence: ++eventSequence,
      at: new Date().toISOString(),
      phase,
      message,
    };
    options.onEvent?.(event);
    if (options.writeOutput !== false) {
      process.stdout.write(`[${phase}] ${message}\n`);
    }
  };

  const cleanup = createIdempotentDevelopmentCleanup([
    {
      name: "stop-timeout-and-signals",
      run: () => {
        clearTimeout(gateTimeout);
        clearInterval(diagnosticTimer);
        signalBinding?.dispose();
        options.signal?.removeEventListener("abort", forwardExternalAbort);
        controller.abort(new Error("development_gate_cleanup"));
      },
    },
    {
      name: "seal-stage-agents",
      run: () => {
        // Prevent a preparation that was already awaiting Python from
        // publishing a late process after the rest of the gate has drained.
        for (const agent of launchAgents) agent.seal();
      },
    },
    {
      name: "deactivate-model",
      run: async () => {
        await activationManager?.close();
      },
    },
    {
      name: "stop-source-workers",
      run: async () => {
        await settleOrThrow(
          workers.map((worker) => worker.stop()),
          "development_worker_stop_failed",
        );
      },
    },
    {
      name: "stop-stage-processes",
      run: async () => {
        await settleOrThrow(
          launchAgents.map((agent) => agent.close()),
          "development_stage_cleanup_failed",
        );
        const active = launchAgents.reduce(
          (total, agent) => total + agent.activeProcessCount,
          0,
        );
        if (active !== 0) {
          throw new Error(`development_stage_processes_remain:${active}`);
        }
      },
    },
    {
      name: "await-source-workers",
      run: async () => {
        // A worker run rejects when the gate itself fails or is cancelled.
        // Await it to avoid orphaned promises, but do not relabel that original
        // runtime failure as a cleanup failure.
        await Promise.all(workerRuns.map((run) => run.catch(() => undefined)));
      },
    },
    {
      name: "close-coordinator",
      run: async () => {
        await coordinator?.close();
      },
    },
    {
      name: "remove-temporary-root",
      run: async () => {
        await removeExpectedTemporaryRoot(temporaryRoot);
      },
    },
  ]);

  if (options.installSignalHandlers !== false) {
    signalBinding = installDevelopmentSignalHandlers(process);
    void signalBinding.signal.then(async (signal) => {
      interruptedSignal = signal;
      controller.abort(new Error(`development_gate_interrupted:${signal}`));
      try {
        await cleanup();
      } finally {
        process.exitCode = signal === "SIGINT" ? 130 : 143;
      }
    }).catch((error) => {
      process.stderr.write(`${renderError(error)}\n`);
      process.exitCode = 1;
    });
  }

  try {
    const ports = await reserveDistinctPorts(3);
    const coordinatorPort = ports[0]!;
    const apiPort = ports[1]!;
    const returnPort = ports[2]!;
    const coordinatorUrl = `http://127.0.0.1:${coordinatorPort}`;
    const networkToken = randomBytes(32).toString("base64url");
    runtimeEnvironment.MYCELLIOS_DEV_E2E_NETWORK_TOKEN = networkToken;
    const authorizedHeaders = {
      authorization: `Bearer ${networkToken}`,
    };
    const runtimeMetadata = readNativeRuntimeBuildMetadata(workspace);
    const config = developmentDistributionConfig({
      modelSource: cli.modelSource,
      coordinatorUrl,
      apiPort,
      returnPort,
      pythonExecutable,
      pythonSource,
      hfHome: runtimeEnvironment.HF_HOME!,
      artifactsDirectory: join(temporaryRoot, "artifacts"),
    });
    let resolveActivated!: (result: AutoDistributionRunResult) => void;
    let rejectActivated!: (error: unknown) => void;
    const activated = new Promise<AutoDistributionRunResult>((resolvePromise, rejectPromise) => {
      resolveActivated = resolvePromise;
      rejectActivated = rejectPromise;
    });

    coordinator = await createCoordinator({
      host: "127.0.0.1",
      port: coordinatorPort,
      databasePath: join(temporaryRoot, "mesh.db"),
      requestTimeoutMs: cli.timeoutMs,
      allowDevelopmentAdapters: true,
      apiAccessEnabled: false,
      networkToken,
    }, {
      logger: environment.MYCELLIOS_DEV_E2E_DEBUG === "1",
      runtimeMetadata,
      ...(cli.modelSource === DEFAULT_MODEL
        ? { modelCapacityInspector: async (input) => developmentTinyModelProfile(input) }
        : {}),
      activationManagerFactory: ({ store, hub, deploymentController }) => {
        const manager = new DynamicModelActivationManager({
          cwd: workspace,
          environment: runtimeEnvironment,
          workerAgentVersion: runtimeMetadata.version,
          ...(runtimeMetadata.buildIdentity
            ? { workerBuildIdentity: runtimeMetadata.buildIdentity }
            : {}),
          snapshot: () => ({
            capacityNodes: config.nodes.map((node) => ({
              id: node.id,
              availableVramMiB: node.memoryMiB - node.reserveMiB,
            })),
            config,
          }),
          resolveManagedAgent: (nodeId, launch) =>
            resolveConnectedExecutorAgent(
              store.listWorkers(),
              hub.connectedWorkerIds(),
              hub,
              nodeId,
              launch,
            ),
          loadProgress: (modelId) => store.listActivationEvents(modelId),
          onProgress: (modelId, event) => {
            store.appendActivationEvent(modelId, event);
            emitPhase(event.phase, event.message);
            if (event.phase === "running_canary") {
              const operation = deploymentController.activeOperationForModel(modelId);
              if (operation) deploymentController.markCanary(operation.id);
            }
            if (event.phase === "failed") {
              rejectActivated(new Error(
                [event.message, ...(event.details ?? [])].join("\n"),
              ));
            }
          },
          onPlanPrepared: (modelId, stages) => {
            const operation = deploymentController.activeOperationForModel(modelId);
            if (!operation) throw new Error(`deployment_operation_missing:${modelId}`);
            const reservation = deploymentController.prepareRoute(
              operation.id,
              stages,
              Math.max(90_000, cli.timeoutMs + 60_000),
            );
            return { id: reservation.id, generation: reservation.generation };
          },
          onPlanHeartbeat: (_modelId, reservationId) =>
            deploymentController.renewRoute(
              reservationId,
              Math.max(90_000, cli.timeoutMs + 60_000),
            ),
          onActivated: (modelId, reservationId, result) => {
            const canary = {
              passed: true,
              text: result.canaryText,
              ...result.canaryMetrics,
              workerId: result.workerId,
            };
            if (reservationId) {
              deploymentController.commitRoute(reservationId, canary);
            } else {
              const operation = deploymentController.activeOperationForModel(modelId);
              if (operation) {
                deploymentController.completeOperation(operation.id, "active", { canary });
              }
            }
            activatedResult = result;
            resolveActivated(result);
          },
        });
        activationManager = manager;
        return manager;
      },
    });
    await coordinator.app.listen({ host: "127.0.0.1", port: coordinatorPort });
    emitPhase("coordinator", `private coordinator ready on ${coordinatorUrl}`);

    const enrollments = new NodeEnrollmentStore(coordinator.database);
    for (const [index, nodeId] of ["dev-node-a", "dev-node-b"].entries()) {
      // The private gate must satisfy the same ownership/key binding as a
      // real node. These enrollments exist only in its temporary database.
      const signer = workerAdmissionSigner(generateWorkerAdmissionCredential());
      const accountId = "development-gate";
      const enrollment = enrollments.issue({
        accountId,
        actor: { kind: "account", id: accountId, scopes: ["node:identity"] },
        expiresInSeconds: 60,
      });
      enrollments.confirm({ enrollmentId: enrollment.enrollmentId, accountId, actorId: accountId });
      const consumed = enrollments.consume({
        enrollmentToken: enrollment.enrollmentToken,
        nonce: enrollment.nonce,
        identityKind: "device",
        identityId: nodeId,
        publicKeyFingerprint: workerAdmissionPublicKeyFingerprint(signer.publicKey) as `sha256:${string}`,
      });
      if (consumed.state !== "consumed") {
        throw new Error(`development_worker_enrollment_failed:${nodeId}:${consumed.state}`);
      }
      const nodeCache = join(cacheRoot, index === 0 ? "node-a" : "node-b");
      const launchAgent = new DevelopmentStageLaunchAgent({
        nodeId,
        pythonExecutable,
        pythonSource,
        cacheDirectory: nodeCache,
        workspace,
        temporaryRoot: join(temporaryRoot, nodeId),
        environment: runtimeEnvironment,
        onProgress: (event) => {
          emitPhase(
            "artifact",
            `${nodeId} layers ${event.layerStart}-${event.layerEnd - 1}: ${event.state}`,
          );
        },
      });
      launchAgents.push(launchAgent);
      const worker = new WorkerAgent(developmentWorkerConfig(index), {
        coordinatorUrl,
        identity: { kind: "device", id: nodeId },
        admissionSigner: signer,
        reconnect: false,
        heartbeatIntervalMs: 1_000,
        advertiseDeployment: false,
        agentVersion: runtimeMetadata.version,
        ...(runtimeMetadata.buildIdentity
          ? { buildIdentity: runtimeMetadata.buildIdentity }
          : {}),
        hardwareCapacityOverride: {
          id: `cpu-${index}`,
          vendor: "CPU",
          model: `Mycellios development CPU ${index + 1}`,
          physicalVramMb: 4_096,
        },
        // The source gate measures a logical CPU route, not host telemetry.
        // Re-probing Windows GPU inventory on every short test heartbeat can
        // overlap hundreds of OS probes while Python prepares model artifacts,
        // starving the control socket and creating a false worker disconnect.
        hardwareProbe: async () => ({
          hostname: nodeId,
          platform: process.platform,
          ramMb: 8_192,
          gpus: [{
            id: `cpu-${index}`,
            vendor: "CPU",
            model: `Mycellios development CPU ${index + 1}`,
            physicalVramMb: 4_096,
          }],
        }),
        distributedExecutor: {
          nodeId,
          stageHost: `${nodeId}.relay`,
          stagePort: 9_850 + index,
          pythonExecutable,
          launchAgent,
          computeMode: "cpu-only",
          cpuEligible: true,
          directTransport: {
            enabled: true,
            listenHost: "127.0.0.1",
            candidateHosts: ["127.0.0.1"],
            publicPortMapping: false,
          },
        },
        logger: {
          info: (message) => emitPhase("worker", `${nodeId}: ${message}`),
          warn: (message) => emitPhase("worker-warning", `${nodeId}: ${message}`),
          error: (message) => emitPhase("worker-error", `${nodeId}: ${message}`),
        },
      });
      workers.push(worker);
      workerRuns.push(worker.start());
    }

    await waitUntil(
      () => coordinator!.hub.connectedWorkerIds().size >= 2
        && coordinator!.database.listWorkerAdmissionCredentials().length === 2,
      30_000,
      () => `development_workers_did_not_authenticate:connected=${
        [...coordinator!.hub.connectedWorkerIds()].join(",") || "none"
      }:credentials=${coordinator!.database.listWorkerAdmissionCredentials().length}`,
      controller.signal,
    );
    const admitted = coordinator.database.listWorkerAdmissionCredentials();
    if (
      admitted.length !== 2
      || admitted.some((credential) => credential.status !== "active")
      || new Set(admitted.map((credential) => credential.identityId)).size !== 2
    ) {
      throw new Error("development_worker_signed_admission_is_invalid");
    }
    emitPhase("workers", "two signed source workers connected");
    if (environment.MYCELLIOS_DEV_E2E_DEBUG === "1") {
      let lastTick = Date.now();
      diagnosticTimer = setInterval(() => {
        const now = Date.now();
        const lagMs = Math.max(0, now - lastTick - 5_000);
        lastTick = now;
        const states = coordinator!.store.listWorkers().map((worker) =>
          `${worker.identityId ?? worker.id}:${worker.status}, heartbeat age ${now - worker.lastSeenAt}ms`);
        emitPhase("control-health", `event-loop lag ${lagMs}ms; ${states.join("; ")}`);
      }, 5_000);
    }

    const requestedResponse = await fetch(`${coordinatorUrl}/public/v1/requested-models`, {
      method: "POST",
      headers: {
        ...authorizedHeaders,
        "content-type": "application/json",
      },
      body: JSON.stringify({
        id: config.model.publicName,
        source: config.model.source,
        revision: config.model.revision,
        contextTokens: config.workload.contextTokens,
        minimumNodes: 2,
        autoActivate: true,
      }),
      signal: boundedAbortSignal(controller.signal, 60_000),
    });
    const requestedText = await requestedResponse.text();
    if (!requestedResponse.ok) {
      throw new Error(
        `development_model_request_http_${requestedResponse.status}:${
          requestedText.slice(0, 1_000)
        }`,
      );
    }
    emitPhase("request", `${config.model.publicName} accepted by the coordinator`);
    activatedResult = await withAbort(activated, controller.signal);

    let observedModels: string[] = [];
    await waitUntil(async () => {
      const response = await fetch(`${coordinatorUrl}/v1/models`, {
        headers: authorizedHeaders,
        signal: boundedAbortSignal(controller.signal, 10_000),
      });
      if (!response.ok) return false;
      const body = await response.json() as { data?: Array<{ id?: string }> };
      observedModels = (body.data ?? [])
        .map((model) => model.id)
        .filter((id): id is string => typeof id === "string");
      return observedModels.includes(config.model.publicName);
    }, 60_000, () =>
      `development_model_was_not_published:observed=${
        observedModels.join(",") || "none"
      }`,
      controller.signal,
    );

    const publishingWorker = coordinator.store.listWorkers()
      .find((worker) => worker.id === activatedResult?.workerId);
    const deployment = publishingWorker?.capabilities.deployments
      .find((candidate) => candidate.model === config.model.publicName);
    const executionNodes = new Set(
      deployment?.execution?.stages?.map((stage) => stage.nodeId) ?? [],
    );
    if (
      deployment?.adapter !== "mycellios-pipeline"
      || executionNodes.size !== 2
      || !executionNodes.has("dev-node-a")
      || !executionNodes.has("dev-node-b")
    ) {
      throw new Error(
        `development_published_route_is_not_two_stage_native:${
          JSON.stringify({
            publishingWorkerId: activatedResult.workerId,
            publishingWorkerFound: publishingWorker !== undefined,
            adapter: deployment?.adapter ?? null,
            executionNodes: [...executionNodes],
          })
        }`,
      );
    }
    emitPhase("catalog", `${config.model.publicName} published after a real canary`);

    const completionResponse = await fetch(`${coordinatorUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        ...authorizedHeaders,
        "content-type": "application/json",
        "idempotency-key": `dev-e2e-${Date.now()}`,
      },
      body: JSON.stringify({
        model: config.model.publicName,
        messages: [{ role: "user", content: "Answer with one short word." }],
        temperature: 0,
        max_tokens: 2,
        stream: false,
      }),
      signal: boundedAbortSignal(controller.signal, 60_000),
    });
    const completionText = await completionResponse.text();
    if (!completionResponse.ok) {
      throw new Error(
        `development_completion_http_${completionResponse.status}:${
          completionText.slice(0, 1_000)
        }`,
      );
    }
    const completion = JSON.parse(completionText) as {
      model?: string;
      choices?: Array<{ message?: { content?: string } }>;
      usage?: { completion_tokens?: number };
      x_network?: {
        execution_trace?: {
          stages?: Array<{ nodeId?: string }>;
          boundaries?: Array<{ transport?: string }>;
        } | null;
      };
    };
    const output = completion.choices?.[0]?.message?.content?.trim() ?? "";
    const traceNodes = new Set(
      completion.x_network?.execution_trace?.stages
        ?.map((stage) => stage.nodeId)
        .filter((nodeId): nodeId is string => typeof nodeId === "string")
        ?? [],
    );
    if (
      completion.model !== config.model.publicName
      || output.length === 0
      || !Number.isInteger(completion.usage?.completion_tokens)
      || (completion.usage?.completion_tokens ?? 0) < 1
      || traceNodes.size !== 2
      || !traceNodes.has("dev-node-a")
      || !traceNodes.has("dev-node-b")
    ) {
      throw new Error("development_completion_evidence_is_invalid");
    }

    const directTransports = coordinator.hub.runtimeTransportSnapshot()
      .filter((transport) => transport.mode === "direct");
    if (directTransports.length < 2) {
      throw new Error("development_gate_did_not_exercise_direct_runtime_links");
    }
    if (directTransports.some((transport) => transport.state === "closed")) {
      throw new Error("development_direct_runtime_link_closed_before_completion");
    }

    const result: DevelopmentGateResult = {
      evidenceClass: "cpu-loopback-logical",
      model: config.model.publicName,
      stages: executionNodes.size,
      signedWorkers: admitted.length,
      directRuntimeLinks: directTransports.length,
      completionTokens: completion.usage!.completion_tokens!,
      canaryPassed: true,
      routePassed: true,
    };
    emitPhase(
      "complete",
      "Two signed logical workers completed a real routed generation.",
    );
    if (options.writeOutput !== false) {
      process.stdout.write(
        "\nMYCELLIOS DISTRIBUTED DEVELOPMENT GATE: PASS\n"
        + `  Model: ${result.model}\n`
        + `  Stages: ${result.stages}\n`
        + `  Signed workers: ${result.signedWorkers}\n`
        + `  Direct runtime links: ${result.directRuntimeLinks}\n`
        + `  Canary output: ${JSON.stringify(activatedResult.canaryText)}\n`
        + `  Routed output: ${JSON.stringify(output)}\n`
        + `  Completion tokens: ${result.completionTokens}\n`,
      );
    }
    return result;
  } catch (error) {
    if (interruptedSignal !== null) return null;
    throw error;
  } finally {
    emitPhase("cleanup", "Stopping workers, stages and the private coordinator…");
    try {
      await cleanup();
      emitPhase("cleanup-complete", "Temporary resources were drained.");
    } catch (error) {
      emitPhase("cleanup-failed", renderError(error).slice(0, 500));
      throw error;
    }
  }
}

function developmentTinyModelProfile(input: {
  source: string;
  revision: string | null;
  contextTokens: number;
  minimumNodes: number;
}): HubModelCapacityProfile {
  const adapter = resolveModelAdapterContract("llama", "LlamaForCausalLM");
  if (!adapter || input.source !== DEFAULT_MODEL) {
    throw new Error("development_tiny_model_profile_is_invalid");
  }
  const weightBytes = 2_125_984;
  const kvBytes = 2 * 4 * 64 * 2 * 2 * input.contextTokens;
  const runtimeBytes = input.minimumNodes * 256 * 1024 * 1024;
  const reserveBytes = Math.ceil(weightBytes * 0.1)
    + input.minimumNodes * 64 * 1024 * 1024;
  const requiredVramMiB = Math.ceil(
    (weightBytes + kvBytes + runtimeBytes + reserveBytes) / (1024 * 1024),
  );
  return {
    schema: "mycellios-hub-model-capacity/1",
    adapterId: adapter.id,
    adapterContractId: adapter.adapterContractId,
    adapterRegistryId: MODEL_ADAPTER_REGISTRY_ID,
    adapterEvidenceScope: MODEL_ADAPTER_EVIDENCE_SCOPE,
    compatible: true,
    incompatibilityReason: null,
    architecture: "LlamaForCausalLM",
    modelType: "llama",
    totalLayers: 2,
    hiddenSize: 16,
    weightBytes,
    requiredVramMiB,
    minimumStageVramMiB: Math.max(
      512,
      Math.ceil(requiredVramMiB / input.minimumNodes * 0.55),
    ),
    minimumNodes: input.minimumNodes,
    contextTokens: input.contextTokens,
    source: input.source,
    revision: input.revision ?? "main",
  };
}

interface DevelopmentStageLaunchAgentOptions {
  nodeId: string;
  pythonExecutable: string;
  pythonSource: string;
  cacheDirectory: string;
  workspace: string;
  temporaryRoot: string;
  environment: NodeJS.ProcessEnv;
  onProgress(event: RuntimePreparationProgressEvent): void;
}

class DevelopmentStageLaunchAgent implements LaunchAgent {
  readonly id: string;
  readonly #local: LocalProcessAgent;
  readonly #active = new Set<LaunchProcessHandle>();
  readonly #preparationAbort = new AbortController();
  #closed = false;

  constructor(private readonly options: DevelopmentStageLaunchAgentOptions) {
    this.id = `dev-stage:${options.nodeId}`;
    this.#local = new LocalProcessAgent({
      id: this.id,
      cwd: options.workspace,
      env: options.environment,
      allowedExecutables: [options.pythonExecutable],
      workspaceRoot: options.temporaryRoot,
      maxWorkspaceBytes: 2 * 1024 * 1024 * 1024,
      stopGraceMs: 5_000,
    });
  }

  get activeProcessCount(): number {
    return this.#active.size;
  }

  seal(): void {
    if (this.#closed) return;
    this.#closed = true;
    this.#preparationAbort.abort(new Error("development_stage_agent_is_closed"));
  }

  async prepareRuntime(
    description: PythonPipelineLaunchDescription,
    nodeId: string,
    onProgress?: (event: RuntimePreparationProgressEvent) => void,
  ): Promise<readonly PythonLaunchProcess[]> {
    if (this.#closed) throw new Error("development_stage_agent_is_closed");
    if (nodeId !== this.options.nodeId) {
      throw new Error("development_stage_artifact_node_mismatch");
    }
    const prepared = await prepareNodeStageArtifacts(description, {
      nodeId,
      pythonExecutable: this.options.pythonExecutable,
      cacheDirectory: this.options.cacheDirectory,
      cwd: this.options.workspace,
      environment: this.options.environment,
      signal: this.#preparationAbort.signal,
      onProgress: (event) => {
        this.options.onProgress(event);
        onProgress?.(event);
      },
    });
    if (this.#closed) throw new Error("development_stage_agent_is_closed");
    return prepared;
  }

  async start(
    request: LaunchAgentStartRequest,
    signal: AbortSignal,
  ): Promise<LaunchProcessHandle> {
    if (this.#closed) throw new Error("development_stage_agent_is_closed");
    const handle = await this.#local.start(request, signal);
    if (this.#closed) {
      await handle.stop("development_stage_agent_closed_during_start");
      throw new Error("development_stage_agent_is_closed");
    }
    this.#active.add(handle);
    void handle.exited.then(
      () => this.#active.delete(handle),
      () => this.#active.delete(handle),
    );
    return handle;
  }

  async close(): Promise<void> {
    if (this.#closed && this.#active.size === 0) return;
    this.seal();
    await settleOrThrow(
      [...this.#active].map((handle) => handle.stop("development_gate_cleanup")),
      "development_stage_process_stop_failed",
    );
    this.#active.clear();
  }
}

interface DevelopmentDistributionConfigOptions {
  modelSource: string;
  coordinatorUrl: string;
  apiPort: number;
  returnPort: number;
  pythonExecutable: string;
  pythonSource: string;
  hfHome: string;
  artifactsDirectory: string;
}

function developmentDistributionConfig(
  options: DevelopmentDistributionConfigOptions,
): AutoDistributionConfig {
  return parseAutoDistributionConfig({
    schema: "gdlp-auto-distribute/1",
    model: {
      source: options.modelSource,
      revision: null,
      publicName: "dev-tiny-distributed",
    },
    nodes: [
      developmentNode("dev-node-a", 9_850),
      developmentNode("dev-node-b", 9_851),
    ],
    links: [
      developmentLink("dev-node-a", "dev-node-b"),
      developmentLink("dev-node-b", "dev-node-a"),
    ],
    distribution: {
      minimumStages: 2,
      maximumStages: 2,
      allowLossyActivation: false,
    },
    workload: {
      promptTokens: 16,
      outputTokens: 4,
      contextTokens: 256,
      concurrentSequences: 1,
      minRouteAvailability: 0.9,
      batchWindowMs: 0,
      p95: true,
    },
    runtime: {
      pythonExecutable: options.pythonExecutable,
      stagePythonExecutable: "python",
      pythonPath: options.pythonSource,
      hfHome: options.hfHome,
      apiEndpoint: { host: "127.0.0.1", port: options.apiPort },
      apiAdvertiseHost: "dev-node-a.relay",
      returnEndpoint: { host: "dev-node-a.relay", port: options.returnPort },
      returnBindHost: "127.0.0.1",
      threadsPerStage: 1,
      connectTimeoutSeconds: 60,
      readinessTimeoutMs: 180_000,
      maxOutputTokens: 64,
    },
    canary: {
      prompt: "Reply with one short word.",
      maxTokens: 2,
      timeoutMs: 60_000,
    },
    coordinator: {
      url: options.coordinatorUrl,
      region: "dev-local",
      maxConcurrency: 1,
      networkTokenEnv: "MYCELLIOS_DEV_E2E_NETWORK_TOKEN",
    },
    artifactsDirectory: options.artifactsDirectory,
  });
}

function developmentWorkerConfig(index: number): WorkerConfig {
  return workerConfigSchema.parse({
    region: "dev-local",
    offeredVramMb: 4_096,
    limits: {
      maxConcurrency: 1,
      pauseWhenForeground: false,
    },
    adapter: {
      kind: "mock",
      developmentOnly: true,
      model: `dev-control-${index + 1}`,
      tokensPerSecond: 1,
      ttftMs: 1,
      failureRate: 0,
    },
    deployment: {
      modelDigest: `sha256:dev-control-${index + 1}`,
      peakVramMb: 1,
      contextLimit: 128,
      tokensPerSecond: 1,
      ttftMs: 1,
    },
  });
}

function developmentNode(id: string, port: number) {
  return {
    id,
    region: "dev-local",
    endpoint: { host: `${id}.relay`, port },
    memoryMiB: 4_096,
    reserveMiB: 256,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    powerWatts: 1,
    availability: 0.999,
    agent: { kind: "managed" as const },
  };
}

function developmentLink(from: string, to: string) {
  return {
    from,
    to,
    oneWayLatencyMs: 0.1,
    jitterP95Ms: 0,
    bandwidthMbps: 10_000,
    lossRate: 0,
    availability: 0.999,
  };
}

function parseCliOptions(argv: readonly string[]): DevelopmentGateCliOptions {
  let help = false;
  let modelSource = DEFAULT_MODEL;
  let runtimeRoot: string | undefined;
  let timeoutMs = DEFAULT_TIMEOUT_SECONDS * 1_000;
  const seen = new Set<string>();
  for (let index = 0; index < argv.length; index += 1) {
    const name = argv[index]!;
    if (name === "--help" || name === "-h") {
      help = true;
      continue;
    }
    if (!["--model", "--runtime", "--timeout"].includes(name)) {
      throw new Error(`development_gate_unknown_argument:${name}`);
    }
    if (seen.has(name)) throw new Error(`development_gate_duplicate_argument:${name}`);
    seen.add(name);
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`development_gate_argument_requires_value:${name}`);
    }
    index += 1;
    if (name === "--model") modelSource = value;
    if (name === "--runtime") runtimeRoot = value;
    if (name === "--timeout") timeoutMs = positiveInteger(value) * 1_000;
  }
  return { help, modelSource, ...(runtimeRoot ? { runtimeRoot } : {}), timeoutMs };
}

function helpText(): string {
  return `Run the complete distributed Mycellios development gate.

Usage:
  npm run dev:e2e:distributed -- [options]

Options:
  --model <id>       Hugging Face model (${DEFAULT_MODEL} by default)
  --runtime <path>   Installed Mycellios Python runtime
  --timeout <sec>    Whole-gate timeout in seconds (default: ${DEFAULT_TIMEOUT_SECONDS})

Set MYCELLIOS_DEV_E2E_HF_HOME to use a prewarmed model cache (relative to this checkout or absolute).

The gate starts a private coordinator, two independently signed source workers,
verified native model stages, the real tunnel/direct transport and a real
generation. It never builds, publishes, or deploys a release. Persistent model
caches are reused; temporary processes, databases and workspaces are drained on
success, failure, SIGINT and SIGTERM.`;
}

function defaultRuntimeRoot(
  workspace: string,
  environment: NodeJS.ProcessEnv,
): string {
  if (process.platform === "win32") {
    const appData = environment.APPDATA;
    if (!appData) throw new Error("APPDATA is required on Windows.");
    const installed = join(appData, "mycellios", "distribution-runtime-v4");
    if (existsSync(installed)) return installed;
  }
  const local = join(workspace, "runtime", "distribution-venv");
  if (existsSync(local)) return local;
  return process.platform === "darwin"
    ? join(homedir(), "Library", "Application Support", "mycellios", "distribution-runtime-v4")
    : join(
        environment.XDG_CONFIG_HOME ?? join(homedir(), ".config"),
        "mycellios",
        "distribution-runtime-v4",
      );
}

async function reserveDistinctPorts(count: number): Promise<number[]> {
  const ports: number[] = [];
  while (ports.length < count) {
    const port = await reservePort();
    if (!ports.includes(port)) ports.push(port);
  }
  return ports;
}

async function reservePort(): Promise<number> {
  return await new Promise<number>((resolvePort, reject) => {
    const server = createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("development_port_reservation_failed"));
        return;
      }
      const port = address.port;
      server.close((error) => error ? reject(error) : resolvePort(port));
    });
  });
}

async function waitUntil(
  predicate: () => boolean | Promise<boolean>,
  timeout: number,
  message: string | (() => string),
  signal?: AbortSignal,
): Promise<void> {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (signal?.aborted) throw abortReason(signal);
    if (await predicate()) return;
    await abortableDelay(100, signal);
  }
  throw new Error(typeof message === "function" ? message() : message);
}

function boundedAbortSignal(
  signal: AbortSignal,
  timeoutMs: number,
): AbortSignal {
  return AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]);
}

function abortableDelay(
  milliseconds: number,
  signal?: AbortSignal,
): Promise<void> {
  if (!signal) return delay(milliseconds);
  if (signal.aborted) return Promise.reject(abortReason(signal));
  return new Promise((resolveDelay, rejectDelay) => {
    const timer = setTimeout(finish, milliseconds);
    const abort = () => finish(abortReason(signal));
    function finish(error?: Error): void {
      clearTimeout(timer);
      signal!.removeEventListener("abort", abort);
      if (error) rejectDelay(error);
      else resolveDelay();
    }
    signal.addEventListener("abort", abort, { once: true });
  });
}

function withAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) return Promise.reject(abortReason(signal));
  return new Promise<T>((resolvePromise, reject) => {
    const abort = () => reject(abortReason(signal));
    signal.addEventListener("abort", abort, { once: true });
    void promise.then(resolvePromise, reject).finally(() => {
      signal.removeEventListener("abort", abort);
    });
  });
}

async function settleOrThrow(
  operations: readonly Promise<unknown>[],
  message: string,
): Promise<void> {
  const settled = await Promise.allSettled(operations);
  const errors = settled.flatMap((result) =>
    result.status === "rejected" ? [normalizeError(result.reason, message)] : []
  );
  if (errors.length > 0) throw new AggregateError(errors, message);
}

async function removeExpectedTemporaryRoot(path: string): Promise<void> {
  const resolvedTemporaryRoot = resolve(path);
  const resolvedSystemTemp = resolve(tmpdir());
  if (
    dirname(resolvedTemporaryRoot) !== resolvedSystemTemp
    || !resolvedTemporaryRoot.startsWith(join(resolvedSystemTemp, "mycellios-dev-e2e-"))
  ) {
    throw new Error(`refusing_to_remove_unexpected_development_directory:${resolvedTemporaryRoot}`);
  }
  await rm(resolvedTemporaryRoot, { recursive: true, force: true });
}

function withTimeout<T>(
  operation: Promise<T>,
  timeoutMs: number,
  message: string,
): Promise<T> {
  let timer: NodeJS.Timeout | null = null;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new Error(message)), timeoutMs);
    // A pending Promise does not keep Node alive. Keep this deadline referenced
    // so a stalled cleanup cannot make the CLI exit successfully before drain.
  });
  return Promise.race([operation, timeout]).finally(() => {
    if (timer) clearTimeout(timer);
  });
}

function abortReason(signal: AbortSignal): Error {
  return signal.reason instanceof Error
    ? signal.reason
    : new Error(String(signal.reason ?? "development_gate_aborted"));
}

function positiveInteger(value: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 3_600) {
    throw new Error("timeout must be an integer between 1 and 3600 seconds");
  }
  return parsed;
}

function normalizeError(value: unknown, fallback: string): Error {
  if (value instanceof Error) return value;
  return new Error(`${fallback}:${String(value)}`);
}

function renderError(value: unknown): string {
  if (value instanceof AggregateError) {
    return [
      value.stack ?? value.message,
      ...value.errors.map((error, index) =>
        `cleanup error ${index + 1}: ${renderError(error)}`
      ),
    ].join("\n");
  }
  return value instanceof Error ? value.stack ?? value.message : String(value);
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}
