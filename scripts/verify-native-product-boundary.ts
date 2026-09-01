import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createAdapter } from "../src/adapters/factory.js";
import { workerConfigSchema } from "../src/contracts/schemas.js";

const workspace = resolve(import.meta.dirname, "..");
const removedProductFiles = [
  "src/adapters/ollama.ts",
  "src/adapters/openai-compatible.ts",
  "src/worker/llmfit.ts",
  "config/worker.ollama.example.json",
  "config/worker.llamacpp.example.json",
  "config/worker.llmfit.example.json",
  "config/worker.local-qwen.example.json",
  "config/worker.mesh-llm.example.json",
  "config/worker.parallax-cell.example.json",
  "config/worker.glm-api.example.json",
  "src/desktop",
  "src/renderer",
  "forge.config.ts",
  "vite.main.config.ts",
  "vite.preload.config.ts",
  "vite.renderer.config.ts",
  "tsconfig.desktop.json",
  ".github/workflows/desktop-build.yml",
];
for (const relativePath of removedProductFiles) {
  if (existsSync(resolve(workspace, relativePath))) {
    throw new Error(`legacy_product_route_present:${relativePath}`);
  }
}

const canonicalProductSurfaces = [
  "package.json",
  "docs/openapi.yaml",
  "src/distribution/python-launcher.ts",
  "src/node/main.ts",
  "src/node/service-definition.ts",
];
const externalRuntimeName =
  /(?:\b(?:ollama|llmfit|nakshatra|vllm|c0mpute|llama[ ._-]?cpp|openai[- ]?compatible|compatible(?:\s+con)?\s+openai)\b|leyten\/shard|\bshard (?:daemon|sidecar|runtime)\b|python\s+-m\s+shard)/i;
for (const relativePath of canonicalProductSurfaces) {
  const source = readFileSync(resolve(workspace, relativePath), "utf8");
  if (externalRuntimeName.test(source)) {
    throw new Error(`external_runtime_in_canonical_surface:${relativePath}`);
  }
}

const externalResearchScripts = [
  "scripts/demo-real.ps1",
  "scripts/setup-local-llamacpp.ps1",
  "scripts/run-local-llamacpp.ps1",
  "scripts/setup-nakshatra-stage.ps1",
] as const;
for (const relativePath of externalResearchScripts) {
  const source = readFileSync(resolve(workspace, relativePath), "utf8");
  if (!source.includes("MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH")) {
    throw new Error(`external_research_script_is_not_isolated:${relativePath}`);
  }
}

const packageManifest = JSON.parse(
  readFileSync(resolve(workspace, "package.json"), "utf8"),
) as { scripts?: Record<string, string> };
const externalResearchCommandFragments = [
  ...externalResearchScripts,
  "scripts/salad/",
  "scripts\\salad\\",
  "deploy/salad/",
  "deploy\\salad\\",
] as const;
for (const [name, command] of Object.entries(packageManifest.scripts ?? {})) {
  for (const fragment of externalResearchCommandFragments) {
    if (
      command.includes(fragment)
      || (
        fragment.startsWith("scripts/")
        && command.includes(fragment.slice("scripts/".length))
      )
    ) {
      throw new Error(`external_research_script_exposed_by_package:${name}`);
    }
  }
  if (command.includes("MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH")) {
    throw new Error(`external_research_gate_exposed_by_package:${name}`);
  }
}

const boundaryModelDigest = `sha256:${"a".repeat(64)}`;
const nativeConfig = {
  region: "boundary-check",
  capacityScope: "cell",
  offeredVramMb: 4_096,
  limits: { maxConcurrency: 1, pauseWhenForeground: false },
  adapter: {
    kind: "mycellios-pipeline",
    model: "boundary-model",
    baseUrl: "http://127.0.0.1:8081",
  },
  deployment: {
    modelDigest: boundaryModelDigest,
    activationId: "boundary-activation",
    contextLimit: 4_096,
  },
} as const;
workerConfigSchema.parse(nativeConfig);

for (const kind of ["ollama", "llamacpp", "openai-compatible"]) {
  assertRejected({
    ...nativeConfig,
    adapter: { kind, model: "legacy", baseUrl: "http://127.0.0.1:8081" },
  }, `legacy_adapter_accepted:${kind}`);
}
assertRejected({
  ...nativeConfig,
  adapter: { ...nativeConfig.adapter, baseUrl: "https://inference.example" },
}, "external_pipeline_endpoint_accepted");
assertRejected({
  ...nativeConfig,
  adapter: {
    ...nativeConfig.adapter,
    apiKeyEnv: "EXTERNAL_PROVIDER_KEY",
    allowedHosts: ["inference.example"],
  },
}, "generic_provider_escape_hatch_accepted");
assertRejected({
  ...nativeConfig,
  llmfit: { enabled: true, executable: "llmfit", arguments: [] },
}, "llmfit_command_configuration_accepted");

const persistedLegacyMock = {
  region: "legacy",
  offeredVramMb: 4_096,
  limits: { maxConcurrency: 1, pauseWhenForeground: false },
  adapter: {
    kind: "mock",
    model: "old-connectivity-check",
    tokensPerSecond: 20,
    ttftMs: 10,
    failureRate: 0,
  },
  deployment: {
    modelDigest: "sha256:legacy-mock",
    contextLimit: 4_096,
  },
};
assertRejected(persistedLegacyMock, "persisted_legacy_mock_accepted");

const explicitDevelopmentMock = workerConfigSchema.parse({
  ...persistedLegacyMock,
  adapter: { ...persistedLegacyMock.adapter, developmentOnly: true },
});
const previousNodeEnvironment = process.env.NODE_ENV;
try {
  process.env.NODE_ENV = "production";
  try {
    createAdapter(explicitDevelopmentMock);
    throw new Error("explicit_development_mock_started_in_production");
  } catch (error) {
    if (
      !(error instanceof Error)
      || error.message !== "mock_adapter_is_not_available_in_production"
    ) {
      throw error;
    }
  }
} finally {
  if (previousNodeEnvironment === undefined) delete process.env.NODE_ENV;
  else process.env.NODE_ENV = previousNodeEnvironment;
}

const example = JSON.parse(
  readFileSync(resolve(workspace, "config/worker.example.json"), "utf8"),
) as unknown;
const parsedExample = workerConfigSchema.parse(example);
if (
  parsedExample.adapter.kind !== "mycellios-native"
  && parsedExample.adapter.kind !== "mycellios-pipeline"
) {
  throw new Error("canonical_worker_example_is_not_native");
}

process.stdout.write("Native Mycellios product boundary verified.\n");

function assertRejected(value: unknown, errorCode: string): void {
  if (workerConfigSchema.safeParse(value).success) throw new Error(errorCode);
}
