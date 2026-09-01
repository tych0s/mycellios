import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { createAdapter } from "../src/adapters/factory.js";
import { workerConfigSchema } from "../src/contracts/schemas.js";

const MODEL_DIGEST = `sha256:${"c".repeat(64)}`;
const ACTIVATION_ID = "native-boundary-activation";
// Legacy external configuration is asserted only as rejected input.
describe("native worker configuration boundary", () => {
  const pipeline = {
    region: "test",
    capacityScope: "cell",
    offeredVramMb: 4_096,
    limits: { maxConcurrency: 1, pauseWhenForeground: false },
    adapter: {
      kind: "mycellios-pipeline",
      model: "model",
      baseUrl: "http://127.0.0.1:8081",
    },
    deployment: {
      modelDigest: MODEL_DIGEST,
      activationId: ACTIVATION_ID,
      contextLimit: 4_096,
    },
  } as const;

  it("accepts only the local native pipeline production adapter", () => {
    expect(workerConfigSchema.safeParse(pipeline).success).toBe(true);
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      adapter: { ...pipeline.adapter, baseUrl: "https://inference.example" },
    }).success).toBe(false);
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      adapter: { ...pipeline.adapter, baseUrl: "http://127.0.0.1:8081/provider" },
    }).success).toBe(false);
  });

  it.each(["ollama", "llamacpp", "openai-compatible"])(
    "rejects legacy external adapter kind %s",
    (kind) => {
      expect(workerConfigSchema.safeParse({
        ...pipeline,
        adapter: {
          kind,
          model: "model",
          baseUrl: "http://127.0.0.1:8081",
        },
      }).success).toBe(false);
    },
  );

  it("rejects llmfit command execution and generic provider escape hatches", () => {
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      llmfit: {
        enabled: true,
        executable: "llmfit",
        arguments: [],
      },
    }).success).toBe(false);
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      adapter: {
        ...pipeline.adapter,
        apiKeyEnv: "PROVIDER_KEY",
        allowedHosts: ["inference.example"],
      },
    }).success).toBe(false);
  });

  it("marks the mock adapter as development-only in the canonical schema", () => {
    const mock = {
      region: "test",
      offeredVramMb: 4_096,
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      adapter: {
        kind: "mock",
        model: "fixture",
        tokensPerSecond: 20,
        ttftMs: 10,
        failureRate: 0,
      },
      deployment: {
        modelDigest: "sha256:fixture",
        contextLimit: 4_096,
      },
    };
    expect(workerConfigSchema.safeParse(mock).success).toBe(false);
    expect(workerConfigSchema.safeParse({
      ...mock,
      adapter: { ...mock.adapter, developmentOnly: true },
    }).success).toBe(true);
  });

  it("cannot start an explicitly persisted development mock in production", () => {
    const config = workerConfigSchema.parse({
      region: "test",
      offeredVramMb: 4_096,
      limits: { maxConcurrency: 1, pauseWhenForeground: false },
      adapter: {
        kind: "mock",
        developmentOnly: true,
        model: "fixture",
        tokensPerSecond: 20,
        ttftMs: 10,
        failureRate: 0,
      },
      deployment: {
        modelDigest: "sha256:fixture",
        contextLimit: 4_096,
      },
    });
    const previous = process.env.NODE_ENV;
    try {
      process.env.NODE_ENV = "production";
      expect(() => createAdapter(config)).toThrow(
        "mock_adapter_is_not_available_in_production",
      );
    } finally {
      if (previous === undefined) delete process.env.NODE_ENV;
      else process.env.NODE_ENV = previous;
    }
  });

  it("requires pinned runtime identities and rejects worker-declared performance", () => {
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      deployment: {
        contextLimit: 4_096,
        activationId: ACTIVATION_ID,
      },
    }).success).toBe(false);
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      deployment: {
        modelDigest: MODEL_DIGEST,
        activationId: ACTIVATION_ID,
        contextLimit: 4_096,
      },
    }).success).toBe(true);
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      deployment: {
        modelDigest: MODEL_DIGEST,
        activationId: ACTIVATION_ID,
        contextLimit: 4_096,
        canaryEvidence: { workerDeclared: true },
        tokensPerSecond: 1_000_000,
        ttftMs: 0,
      },
    }).success).toBe(false);
    expect(workerConfigSchema.safeParse({
      ...pipeline,
      deployment: {
        ...pipeline.deployment,
        activationId: undefined,
      },
    }).success).toBe(false);
  });

  it("keeps canonical product surfaces free of external runtime names", () => {
    const externalRuntimeName =
      /(?:\b(?:ollama|llmfit|nakshatra|vllm|c0mpute|llama[ ._-]?cpp|openai[- ]?compatible|compatible(?:\s+con)?\s+openai)\b|leyten\/shard|\bshard (?:daemon|sidecar|runtime)\b|python\s+-m\s+shard)/i;
    for (const relativePath of [
      "package.json",
      "docs/openapi.yaml",
      "src/distribution/python-launcher.ts",
    ]) {
      expect(
        readFileSync(resolve(process.cwd(), relativePath), "utf8"),
        relativePath,
      ).not.toMatch(externalRuntimeName);
    }
  });

  it("keeps archived external-runtime scripts behind an explicit research gate", () => {
    const externalResearchScripts = [
      "scripts/demo-real.ps1",
      "scripts/setup-local-llamacpp.ps1",
      "scripts/run-local-llamacpp.ps1",
      "scripts/setup-nakshatra-stage.ps1",
    ] as const;
    for (const relativePath of externalResearchScripts) {
      expect(
        readFileSync(resolve(process.cwd(), relativePath), "utf8"),
        relativePath,
      ).toContain("MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH");
    }

    const packageManifest = JSON.parse(
      readFileSync(resolve(process.cwd(), "package.json"), "utf8"),
    ) as { scripts?: Record<string, string> };
    const externalResearchCommandFragments = [
      ...externalResearchScripts,
      "scripts/salad/",
      "scripts\\salad\\",
      "deploy/salad/",
      "deploy\\salad\\",
    ] as const;
    for (const [name, command] of Object.entries(packageManifest.scripts ?? {})) {
      expect(command, name).not.toContain("MYCELLIOS_EXTERNAL_RUNTIME_RESEARCH");
      for (const fragment of externalResearchCommandFragments) {
        expect(command, name).not.toContain(fragment);
        if (fragment.startsWith("scripts/")) {
          expect(command, name).not.toContain(fragment.slice("scripts/".length));
        }
      }
    }
  });
});
