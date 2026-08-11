import { z } from "zod";
import {
  coordinatorRuntimePerformanceEvidenceSchema,
} from "../performance/runtime-profile.js";
import {
  deploymentCanaryEvidenceSchema,
  deploymentMetricsFromCanaryEvidence,
} from "./deployment-canary.js";
import { nativeBuildIdentitySchema } from "./build-identity.js";
import {
  workerAdmissionProofSchema,
  workerProtocolRangeSchema,
} from "./worker-admission.js";

const adapterKind = z.enum([
  "mycellios-native",
  "mycellios-pipeline",
  "mock",
]);
const deploymentAdapterKind = z.enum(["mycellios-pipeline", "mock"]);
const executionBackend = z.enum(["cpu", "cuda", "rocm", "directml", "mps", "xpu", "vulkan", "webgpu"]);
const sha256Digest = z.string().regex(/^sha256:[0-9a-f]{64}$/);

const executionStageSchema = z.object({
  nodeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
  stageIndex: z.number().int().nonnegative(),
  layerStart: z.number().int().nonnegative(),
  layerEnd: z.number().int().positive(),
  deviceType: z.enum(["cpu", "gpu"]),
  backend: executionBackend,
  deviceName: z.string().min(1).max(256),
  precision: z.string().min(1).max(64),
  fallback: z.boolean(),
  fallbackReason: z.string().min(1).max(1_024).optional(),
}).strict().superRefine((stage, context) => {
  if (stage.layerEnd <= stage.layerStart) {
    context.addIssue({ code: "custom", message: "layerEnd must be greater than layerStart", path: ["layerEnd"] });
  }
  if ((stage.deviceType === "cpu") !== (stage.backend === "cpu")) {
    context.addIssue({ code: "custom", message: "CPU stages require the cpu backend and GPU stages require a GPU backend", path: ["backend"] });
  }
  if (!stage.fallback && stage.fallbackReason !== undefined) {
    context.addIssue({ code: "custom", message: "fallbackReason requires fallback=true", path: ["fallbackReason"] });
  }
});

const executionTelemetrySchema = z.object({
  deviceType: z.enum(["cpu", "gpu", "mixed"]),
  backend: executionBackend,
  deviceName: z.string().min(1).max(256),
  precision: z.string().min(1).max(64),
  fallback: z.boolean(),
  fallbackReason: z.string().min(1).max(1_024).optional(),
  stages: z.array(executionStageSchema).min(1).max(64).optional(),
}).strict().superRefine((execution, context) => {
  if (execution.deviceType === "cpu" && execution.backend !== "cpu") {
    context.addIssue({ code: "custom", message: "CPU execution requires the cpu backend", path: ["backend"] });
  }
  if (execution.deviceType === "gpu" && execution.backend === "cpu") {
    context.addIssue({ code: "custom", message: "GPU execution requires a GPU backend", path: ["backend"] });
  }
  if (!execution.fallback && execution.fallbackReason !== undefined) {
    context.addIssue({ code: "custom", message: "fallbackReason requires fallback=true", path: ["fallbackReason"] });
  }
  if (!execution.stages) return;
  const stageIndexes = new Set<number>();
  for (const [index, stage] of execution.stages.entries()) {
    if (stageIndexes.has(stage.stageIndex)) {
      context.addIssue({ code: "custom", message: "stageIndex values must be unique", path: ["stages", index, "stageIndex"] });
    }
    stageIndexes.add(stage.stageIndex);
  }
  const hasCpu = execution.stages.some((stage) => stage.deviceType === "cpu");
  const hasGpu = execution.stages.some((stage) => stage.deviceType === "gpu");
  const derived = hasCpu && hasGpu ? "mixed" : hasGpu ? "gpu" : "cpu";
  if (execution.deviceType !== derived) {
    context.addIssue({ code: "custom", message: "deviceType must match the effective stage devices", path: ["deviceType"] });
  }
});

export const chatMessageSchema = z.object({
  role: z.enum(["system", "developer", "user", "assistant", "tool"]),
  content: z.string().min(1),
  name: z.string().min(1).optional(),
});

export const chatCompletionRequestSchema = z.object({
  model: z.string().min(1),
  messages: z.array(chatMessageSchema).min(1),
  stream: z.boolean().default(false),
  max_tokens: z.number().int().min(1).max(32_768).default(256),
  temperature: z.number().min(0).max(2).default(0.7),
  top_p: z.number().gt(0).max(1).default(1),
  seed: z.number().int().optional(),
  session_id: z.string().min(1).optional(),
  workload_class: z.enum(["interactive", "batch", "benchmark"]).default("interactive"),
  preferred_region: z.string().min(1).optional(),
  deadline_ms: z.number().int().min(1_000).max(3_600_000).default(120_000),
  privacy: z.object({
    trust: z.enum(["default", "trusted-only"]).default("default"),
    boundary: z.enum(["trusted-edges", "pinned-edges"]).default("trusted-edges"),
    pinned_identity_ids: z.array(z.string().min(1).max(128).regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/)).min(1).max(64).optional(),
  }).strict().superRefine((privacy, context) => {
    if (privacy.boundary === "pinned-edges" && !privacy.pinned_identity_ids) {
      context.addIssue({ code: "custom", path: ["pinned_identity_ids"], message: "pinned-edges requires explicit trusted identities" });
    }
    if (privacy.boundary !== "pinned-edges" && privacy.pinned_identity_ids) {
      context.addIssue({ code: "custom", path: ["pinned_identity_ids"], message: "Pinned identities require pinned-edges" });
    }
  }).optional(),
}).strict();

export const deploymentSchema = z
  .object({
    deploymentId: z.string().min(1),
    model: z.string().min(1),
    modelDigest: z.string().min(1),
    activationId: z.string().trim().min(1).max(256).optional(),
    mode: z.enum(["replica", "pipeline"]),
    adapter: deploymentAdapterKind,
    peakVramMb: z.number().int().positive(),
    contextLimit: z.number().int().positive(),
    maxConcurrency: z.number().int().positive(),
    freeSlots: z.number().int().nonnegative(),
    tokensPerSecond: z.number().positive(),
    throughputSource: z.enum(["measured", "estimated", "configured", "default"]).optional(),
    ttftMs: z.number().nonnegative(),
    verificationState: z.enum(["pending", "verified"]).optional(),
    canaryEvidence: deploymentCanaryEvidenceSchema.optional(),
    dataLocality: z.literal("local"),
    stage: z
      .object({
        index: z.number().int().nonnegative(),
        total: z.number().int().positive(),
        layerStart: z.number().int().nonnegative(),
        layerEnd: z.number().int().nonnegative(),
      })
      .optional(),
    internalPipeline: z
      .object({
        stageCount: z.number().int().min(2).max(64),
        boundaries: z.array(z.number().int().nonnegative()).min(3).max(65),
      })
      .strict()
      .optional(),
    execution: executionTelemetrySchema.optional(),
  })
  .superRefine((deployment, context) => {
    if (deployment.mode === "pipeline" && deployment.stage === undefined) {
      context.addIssue({
        code: "custom",
        message: "A pipeline deployment requires stage metadata",
        path: ["stage"],
      });
    }
    if (deployment.stage && deployment.stage.index >= deployment.stage.total) {
      context.addIssue({
        code: "custom",
        message: "stage.index must be lower than stage.total",
        path: ["stage", "index"],
      });
    }
    if (deployment.internalPipeline) {
      const { stageCount, boundaries } = deployment.internalPipeline;
      if (
        boundaries.length !== stageCount + 1 ||
        boundaries[0] !== 0 ||
        boundaries.some((boundary, index) => index > 0 && boundary <= boundaries[index - 1]!)
      ) {
        context.addIssue({
          code: "custom",
          message: "internalPipeline boundaries must be strictly increasing and match stageCount",
          path: ["internalPipeline", "boundaries"],
        });
      }
    }
    if (deployment.adapter === "mycellios-pipeline") {
      if (
        !deployment.activationId
        || !deployment.verificationState
      ) {
        context.addIssue({
          code: "custom",
          message: "A native pipeline deployment requires an explicit verification state",
          path: ["verificationState"],
        });
      } else if (deployment.verificationState === "pending") {
        if (
          deployment.canaryEvidence
          || deployment.throughputSource !== "default"
        ) {
          context.addIssue({
            code: "custom",
            message: "A pending native pipeline cannot publish measured evidence",
            path: ["canaryEvidence"],
          });
        }
      } else if (
        !deployment.canaryEvidence
        || deployment.throughputSource !== "measured"
      ) {
        context.addIssue({
          code: "custom",
          message: "A verified native pipeline requires coordinator-observed canary evidence",
          path: ["canaryEvidence"],
        });
      } else {
        try {
          const measured = deploymentMetricsFromCanaryEvidence(
            deployment.canaryEvidence,
            {
              model: deployment.model,
              modelDigest: deployment.modelDigest,
              activationId: deployment.activationId,
            },
          );
          if (
            measured.tokensPerSecond !== deployment.tokensPerSecond
            || measured.ttftMs !== deployment.ttftMs
          ) {
            context.addIssue({
              code: "custom",
              message: "Native pipeline metrics must be derived from its canary evidence",
              path: ["tokensPerSecond"],
            });
          }
        } catch (error) {
          context.addIssue({
            code: "custom",
            message: error instanceof Error ? error.message : String(error),
            path: ["canaryEvidence"],
          });
        }
      }
    } else if (
      deployment.canaryEvidence
      || deployment.activationId
      || deployment.verificationState
    ) {
      context.addIssue({
        code: "custom",
        message: "Canary activation evidence is reserved for native pipelines",
        path: ["canaryEvidence"],
      });
    }
  });

export const gpuSchema = z
  .object({
    id: z.string().min(1),
    vendor: z.string().min(1),
    model: z.string().min(1),
    physicalVramMb: z.number().int().nonnegative(),
    sharedMemoryMb: z.number().int().nonnegative().optional(),
    unifiedMemory: z.boolean().optional(),
    offeredVramMb: z.number().int().min(512),
    freeOfferedVramMb: z.number().int().nonnegative(),
    utilizationPct: z.number().min(0).max(100).optional(),
    temperatureC: z.number().optional(),
    powerW: z.number().nonnegative().optional(),
  })
  .superRefine((gpu, context) => {
    const detectedBudget = gpu.physicalVramMb + (gpu.sharedMemoryMb ?? 0);
    if (detectedBudget > 0 && gpu.offeredVramMb > detectedBudget) {
      context.addIssue({
        code: "custom",
        message: "offeredVramMb cannot exceed detected dedicated plus shared GPU memory",
        path: ["offeredVramMb"],
      });
    }
    if (gpu.freeOfferedVramMb > gpu.offeredVramMb) {
      context.addIssue({
        code: "custom",
        message: "freeOfferedVramMb cannot exceed the offered budget",
        path: ["freeOfferedVramMb"],
      });
    }
  });

export const workerCapabilitiesSchema = z.object({
  region: z.string().min(1),
  agentVersion: z.string().min(1),
  buildIdentity: nativeBuildIdentitySchema.optional(),
  administration: z
    .object({
      contributionControl: z.literal("mycellios-contribution-control/1"),
    })
    .strict()
    .optional(),
  gpus: z.array(gpuSchema).min(1),
  limits: z.object({
    maxConcurrency: z.number().int().positive(),
    maxPowerW: z.number().positive().optional(),
    maxTemperatureC: z.number().optional(),
    pauseWhenForeground: z.boolean(),
  }),
  // A contributor may register real hardware before it has a real inference
  // runtime available. The scheduler only considers workers that advertise a
  // matching deployment, so an empty list is safe and avoids inventing one.
  deployments: z.array(deploymentSchema).max(256),
  network: z.object({
    coordinatorRttMs: z.number().nonnegative(),
    uplinkMbps: z.number().nonnegative(),
    downlinkMbps: z.number().nonnegative(),
  }),
  distributedExecutor: z
    .object({
      protocol: z.enum(["gdlp-worker-tunnel/1", "gdlp-worker-tunnel/2"]),
      streamRecovery: z.literal("offset-ack-v1").optional(),
      nodeId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
      stageHost: z.string().min(1).max(253),
      stagePort: z.number().int().min(1).max(65_535),
      runtime: z.literal("python-safetensors"),
      computeMode: z.enum(["automatic", "gpu-only", "cpu-only"]).default("automatic"),
      // Legacy clients must never gain CPU scheduling consent merely by
      // upgrading the coordinator; only a new client can opt in explicitly.
      cpuEligible: z.boolean().default(false),
      acceleration: z
        .object({
          schema: z.literal("mycellios-accelerator-diagnostics/1"),
          appVersion: z.string().min(1).max(64),
          state: z.enum(["idle", "preparing", "cpu-ready", "gpu-ready", "gpu-fallback", "error"]),
          backend: z.enum(["cpu", "cuda", "rocm", "mps", "xpu"]).nullable(),
          deviceName: z.string().max(256).nullable(),
          gpuVendor: z.string().max(128).nullable(),
          gpuModel: z.string().max(256).nullable(),
          phase: z.enum([
            "idle",
            "detecting",
            "checking-cache",
            "checking-prerequisites",
            "copying-base",
            "downloading",
            "verifying-package",
            "installing",
            "physical-probe",
            "activating",
            "ready",
            "fallback",
            "blocked",
            "error",
          ]),
          progressPct: z.number().min(0).max(100).nullable(),
          issueCode: z.string().max(64).nullable(),
          issueSummary: z.string().max(500).nullable(),
          retryable: z.boolean(),
          retryAttempt: z.number().int().nonnegative().max(1_000_000),
          nextRetryAt: z.string().datetime().nullable(),
          updatedAt: z.string().datetime(),
          recentEvents: z.array(z.object({
            at: z.string().datetime(),
            level: z.enum(["info", "success", "warning", "error"]),
            message: z.string().min(1).max(500),
          }).strict()).max(12),
        })
        .strict()
        .optional(),
      /**
       * Coordinator-observed calibration produced only after a directed probe
       * challenge on the current authenticated worker session.
       */
      performanceEvidence: coordinatorRuntimePerformanceEvidenceSchema.optional(),
      isolation: z
        .object({
          schema: z.literal("mycellios-executor-isolation-capability/1"),
          launchPolicySchema: z.literal("gdlp-executor-isolation/4"),
          environment: z.literal("filtered"),
          workspace: z.literal("private-temp-watchdog"),
          processTree: z.enum(["best-effort", "windows-job-object"]),
          resourceLimits: z.literal("workspace-watchdog-only"),
          osSandbox: z.literal("not-enforced"),
          hardResourceQuotas: z.literal("not-enforced"),
          killOnClose: z.enum(["not-enforced", "windows-job-object"]),
          maxWorkspaceBytes: z.number().int().min(64 * 1024).max(64 * 1024 * 1024 * 1024),
          maxWorkspaceEntries: z.number().int().min(16).max(1_000_000),
          workspaceCheckIntervalMs: z.number().int().min(25).max(60_000),
        })
        .strict()
        .refine(
          (value) =>
            (value.processTree === "best-effort" && value.killOnClose === "not-enforced")
            || (
              value.processTree === "windows-job-object"
              && value.killOnClose === "windows-job-object"
            ),
          { message: "executor isolation process-tree guarantees are inconsistent" },
        )
        .optional(),
      directTransport: z
        .object({
          protocol: z.literal("mycellios-direct/1"),
          commitAck: z.literal("destination-v1").optional(),
          candidates: z.array(z.object({
            host: z.string().min(1).max(253),
            port: z.number().int().min(1).max(65_535),
            scope: z.enum(["lan", "configured", "public-mapped"]),
          }).strict()).min(1).max(8),
          maxSessions: z.number().int().min(1).max(256),
          maxSessionBytes: z.number().int().min(1024 * 1024).max(16 * 1024 * 1024 * 1024),
        })
        .strict()
        .optional(),
      physicalIdentity: z
        .object({
          schema: z.literal("gdlp-worker-physical-identity/1"),
          provider: z.enum(["gpu_cloud", "generic"]),
          providerMachineFingerprintSha256: sha256Digest,
          hostFingerprintSha256: sha256Digest,
          gpuFingerprintsSha256: z.array(sha256Digest).min(1).max(64),
          attestedAt: z.iso.datetime(),
        })
        .strict()
        .optional(),
    })
      .strict()
      .optional(),
}).strict().superRefine((capabilities, context) => {
  if (
    capabilities.buildIdentity
    && capabilities.buildIdentity.version !== capabilities.agentVersion
  ) {
    context.addIssue({
      code: "custom",
      path: ["buildIdentity", "version"],
      message: "buildIdentity.version must match agentVersion",
    });
  }
});

export const workerRegistrationSchema = z.object({
  identity: z
    .object({
      kind: z.enum(["device", "cell"]),
      id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
    })
    .strict()
    .optional(),
  capabilities: workerCapabilitiesSchema,
  protocol: workerProtocolRangeSchema.optional(),
  admission: workerAdmissionProofSchema.optional(),
}).strict();

export const workerConfigSchema = z.object({
  region: z.string().min(1),
  instanceId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/).optional(),
  // `cell` means that offeredVramMb is the aggregate capacity of a native
  // Mycellios pipeline represented by this gateway, not one physical GPU.
  capacityScope: z.enum(["host", "cell"]).default("host"),
  offeredVramMb: z.number().int().min(512),
  limits: z.object({
    maxConcurrency: z.number().int().positive().default(1),
    maxPowerW: z.number().positive().optional(),
    maxTemperatureC: z.number().optional(),
    pauseWhenForeground: z.boolean().default(true),
  }),
  adapter: z.discriminatedUnion("kind", [
    z.object({
      kind: z.literal("mycellios-native"),
      model: z.literal("mycellios-native-control"),
    }),
    z.object({
      kind: z.literal("mock"),
      developmentOnly: z.literal(true),
      model: z.string().min(1),
      tokensPerSecond: z.number().positive().default(20),
      ttftMs: z.number().nonnegative().default(150),
      failureRate: z.number().min(0).max(1).default(0),
    }),
    z.object({
      kind: z.literal("mycellios-pipeline"),
      model: z.string().min(1),
      baseUrl: z.string().url().superRefine((value, context) => {
        try {
          const url = new URL(value);
          const hostname = url.hostname.toLowerCase();
          if (
            !new Set(["localhost", "127.0.0.1", "::1", "[::1]"]).has(hostname)
            || !new Set(["http:", "https:"]).has(url.protocol)
            || url.username !== ""
            || url.password !== ""
            || url.search !== ""
            || url.hash !== ""
            || !new Set(["", "/"]).has(url.pathname)
          ) {
            context.addIssue({
              code: "custom",
              message: "mycellios-pipeline requires a credential-free loopback origin",
            });
          }
        } catch {
          context.addIssue({
            code: "custom",
            message: "mycellios-pipeline requires a valid loopback origin",
          });
        }
      }),
    }).strict(),
  ]),
  deployment: z
    .object({
      modelDigest: z.string().min(1).optional(),
      activationId: z.string().trim().min(1).max(256).optional(),
      peakVramMb: z.number().int().positive().optional(),
      contextLimit: z.number().int().positive().default(8_192),
      tokensPerSecond: z.number().positive().optional(),
      ttftMs: z.number().nonnegative().optional(),
      internalPipeline: z
        .object({
          stageCount: z.number().int().min(2).max(64),
          boundaries: z.array(z.number().int().nonnegative()).min(3).max(65),
        })
        .strict()
        .optional(),
      execution: executionTelemetrySchema.optional(),
    })
    .default({ contextLimit: 8_192 }),
}).strict().superRefine((config, context) => {
  if (
    config.adapter.kind === "mycellios-pipeline"
    && !config.deployment.modelDigest
  ) {
    context.addIssue({
      code: "custom",
      message: "A real inference adapter requires a pinned modelDigest",
      path: ["deployment", "modelDigest"],
    });
  }
  if (config.capacityScope === "cell" && config.adapter.kind !== "mycellios-pipeline") {
    context.addIssue({
      code: "custom",
      message: "capacityScope=cell requires a native Mycellios pipeline gateway",
      path: ["capacityScope"],
    });
  }
  if (
    config.adapter.kind === "mycellios-pipeline"
    && !config.deployment.activationId
  ) {
    context.addIssue({
      code: "custom",
      message: "mycellios-pipeline requires an independently verified activationId",
      path: ["deployment", "activationId"],
    });
  }
  if (
    config.adapter.kind === "mycellios-pipeline"
    && (
      config.deployment.tokensPerSecond !== undefined
      || config.deployment.ttftMs !== undefined
    )
  ) {
    context.addIssue({
      code: "custom",
      message: "mycellios-pipeline performance cannot be configured manually",
      path: ["deployment"],
    });
  }
});

export type WorkerConfig = z.infer<typeof workerConfigSchema>;
