import { z } from "zod";

const adapterKind = z.enum(["mock", "local-model-runtime", "externalggufruntime", "openai-compatible"]);
const executionBackend = z.enum(["cpu", "cuda", "rocm", "directml", "mps", "xpu", "vulkan", "webgpu"]);

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
}).strict();

export const deploymentSchema = z
  .object({
    deploymentId: z.string().min(1),
    model: z.string().min(1),
    modelDigest: z.string().min(1),
    mode: z.enum(["replica", "pipeline"]),
    adapter: adapterKind,
    peakVramMb: z.number().int().positive(),
    contextLimit: z.number().int().positive(),
    maxConcurrency: z.number().int().positive(),
    freeSlots: z.number().int().nonnegative(),
    tokensPerSecond: z.number().positive(),
    ttftMs: z.number().nonnegative(),
    dataLocality: z.enum(["local", "external"]),
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
  llmfit: z
    .object({
      source: z.literal("llmfit"),
      scope: z.literal("host"),
      backend: z.string().min(1),
      cpuName: z.string(),
      cpuCores: z.number().int().nonnegative(),
      totalRamMb: z.number().int().nonnegative(),
      availableRamMb: z.number().int().nonnegative(),
      gpuCount: z.number().int().nonnegative(),
      gpus: z.array(
        z.object({
          name: z.string().min(1),
          backend: z.string().min(1),
          vramMb: z.number().int().nonnegative(),
          unifiedMemory: z.boolean(),
        }).strict(),
      ),
      model: z
        .object({
          deploymentId: z.string().min(1).optional(),
          requestedModel: z.string().min(1),
          resolvedModel: z.string().min(1),
          fitLevel: z.string().min(1),
          runMode: z.string().min(1),
          runtime: z.string().min(1).optional(),
          bestQuant: z.string().min(1).optional(),
          estimatedTokensPerSecond: z.number().positive().optional(),
          measuredTokensPerSecond: z.number().positive().optional(),
          memoryRequiredMb: z.number().int().nonnegative().optional(),
          usableContext: z.number().int().positive().optional(),
        })
        .strict()
        .optional(),
    })
    .strict()
    .optional(),
  distributedExecutor: z
    .object({
      protocol: z.enum(["gdlp-worker-tunnel/1", "gdlp-worker-tunnel/2"]),
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
    })
    .strict()
    .optional(),
}).strict();

export const workerRegistrationSchema = z.object({
  identity: z
    .object({
      kind: z.enum(["device", "cell"]),
      id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/),
    })
    .strict()
    .optional(),
  capabilities: workerCapabilitiesSchema,
}).strict();

export const workerConfigSchema = z.object({
  region: z.string().min(1),
  instanceId: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/).optional(),
  // `cell` means that offeredVramMb is the aggregate capacity of a sidecar
  // cell represented by this gateway, not the gateway host's physical GPU.
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
      kind: z.literal("mock"),
      model: z.string().min(1),
      tokensPerSecond: z.number().positive().default(20),
      ttftMs: z.number().nonnegative().default(150),
      failureRate: z.number().min(0).max(1).default(0),
    }),
    z.object({
      kind: z.literal("local-model-runtime"),
      model: z.string().min(1),
      baseUrl: z.string().url().default("http://127.0.0.1:11434"),
    }),
    z.object({
      kind: z.literal("externalggufruntime"),
      model: z.string().min(1),
      baseUrl: z.string().url().default("http://127.0.0.1:8080"),
    }),
    z.object({
      kind: z.literal("openai-compatible"),
      model: z.string().min(1),
      baseUrl: z.string().url(),
      requestTemperature: z.number().min(0).max(2).optional(),
      apiPathPrefix: z
        .string()
        .max(128)
        .regex(/^(?:[A-Za-z0-9._~-]+\/?)*$/)
        .default("v1"),
      apiKeyEnv: z.string().min(1).optional(),
      allowedHosts: z.array(z.string().min(1)).default([]),
    }),
  ]),
  deployment: z
    .object({
      modelDigest: z.string().min(1).optional(),
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
  llmfit: z
    .object({
      enabled: z.boolean().default(false),
      required: z.boolean().default(false),
      executable: z.string().min(1).max(4_096).default("llmfit"),
      arguments: z.array(z.string().max(4_096)).max(32).default([]),
      timeoutMs: z.number().int().min(1_000).max(120_000).default(15_000),
      model: z.string().min(1).optional(),
      applyPerformanceEstimate: z.boolean().default(false),
    })
    .strict()
    .default({
      enabled: false,
      required: false,
      executable: "llmfit",
      arguments: [],
      timeoutMs: 15_000,
      applyPerformanceEstimate: false,
    }),
}).strict().superRefine((config, context) => {
  if (config.adapter.kind !== "mock" && !config.deployment.modelDigest) {
    context.addIssue({
      code: "custom",
      message: "A real inference adapter requires a pinned modelDigest",
      path: ["deployment", "modelDigest"],
    });
  }
  if (config.capacityScope === "cell" && config.adapter.kind !== "openai-compatible") {
    context.addIssue({
      code: "custom",
      message: "capacityScope=cell requires an OpenAI-compatible sidecar gateway",
      path: ["capacityScope"],
    });
  }
});

export type WorkerConfig = z.infer<typeof workerConfigSchema>;
