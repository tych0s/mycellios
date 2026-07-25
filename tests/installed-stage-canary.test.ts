import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { MODEL_ADAPTER_REGISTRY_ID } from "../src/contracts/model-adapter-registry.js";
import {
  INSTALLED_STAGE_CANARY_MARKER,
  runInstalledStageCanary,
  validateInstalledStageCanaryEvidence,
} from "../scripts/installed-stage-canary.mjs";

const validEvidence = {
  schema: "mycellios-installed-stage-canary/1",
  ok: true,
  pythonVersion: "3.12.13",
  pythonPrefix: "C:\\runtime",
  torchVersion: "2.13.0+cpu",
  transformersVersion: "5.14.1",
  engine: "python-torch",
  adapter: "transformers-llama-v1",
  adapterContractId: "sha256:baf162a91e7a47c8cef264335c840ed5292c7583968473d952d7867f989f6439",
  adapterRegistryId: MODEL_ADAPTER_REGISTRY_ID,
  loader: "selective-safetensors",
  batchSize: 2,
  physicalBatchCalls: 2,
  physicalBatchItems: 4,
  sequenceTokens: 3,
  kvBytes: 512,
  copiedKvBytes: 512,
  batchQueueBatches: 1,
  outputTokenSha256: "a".repeat(64),
  parity: {
    sequentialVsBatch: true,
    fork: true,
    rollback: true,
  },
};
const adapterRegistrySource = readFileSync(
  "python/distributed_runtime/model_adapter_registry.json",
  "utf8",
);
const acceptProductSource = () => ({});

describe("installed stage canary gate", () => {
  it("accepts only the exact runtime and parity contract", () => {
    expect(() =>
      validateInstalledStageCanaryEvidence(validEvidence, {
        expectedTorchVersion: "2.13.0+cpu",
        expectedTransformersVersion: "5.14.1",
      }),
    ).not.toThrow();
  });

  it("fails closed when numerical parity is missing", () => {
    expect(() =>
      validateInstalledStageCanaryEvidence({
        ...validEvidence,
        parity: { ...validEvidence.parity, rollback: false },
      }),
    ).toThrow("parity evidence is invalid");
  });

  it("fails before spawning when packaged Mycellios source is incomplete", () => {
    const spawnSync = vi.fn();
    expect(() =>
      runInstalledStageCanary(
        {
          pythonExecutable: "C:\\runtime\\python.exe",
          pythonSourceRoot: "C:\\resources\\python",
        },
        {
          existsSync: (path: string) => path.endsWith("python.exe"),
          spawnSync,
        },
      ),
    ).toThrow("source is missing");
    expect(spawnSync).not.toHaveBeenCalled();
  });

  it("fails before spawning when the packaged Python runtime is missing", () => {
    const spawnSync = vi.fn();
    expect(() =>
      runInstalledStageCanary(
        {
          pythonExecutable: "C:\\runtime\\python.exe",
          pythonSourceRoot: "C:\\resources\\python",
        },
        {
          existsSync: () => false,
          spawnSync,
        },
      ),
    ).toThrow("Python is missing");
    expect(spawnSync).not.toHaveBeenCalled();
  });

  it("propagates a real canary process failure instead of accepting its logs", () => {
    expect(() =>
      runInstalledStageCanary(
        {
          pythonExecutable: "C:\\runtime\\python.exe",
          pythonSourceRoot: "C:\\resources\\python",
        },
        {
          existsSync: () => true,
          readFileSync: () => adapterRegistrySource,
          verifyNativePythonProductSource: acceptProductSource,
          spawnSync: () => ({
            status: 1,
            stdout: "looks healthy",
            stderr: "KV parity failed",
          }),
        },
      ),
    ).toThrow("KV parity failed");
  });

  it("imports every native product entrypoint before running parity", () => {
    let processArguments: readonly string[] = [];
    runInstalledStageCanary(
      {
        pythonExecutable: "C:\\runtime\\python.exe",
        pythonSourceRoot: "C:\\resources\\python",
      },
      {
        existsSync: () => true,
        readFileSync: () => adapterRegistrySource,
        verifyNativePythonProductSource: acceptProductSource,
        spawnSync: (_executable, args) => {
          processArguments = args;
          return {
            status: 0,
            stdout:
              INSTALLED_STAGE_CANARY_MARKER + JSON.stringify(validEvidence),
            stderr: "",
          };
        },
      },
    );
    const modules = JSON.parse(processArguments.at(-1) ?? "[]") as string[];
    expect(modules).toEqual(expect.arrayContaining([
      "distributed_runtime.server",
      "distributed_runtime.stage_cli",
      "distributed_runtime.native_gguf_runtime",
      "distributed_runtime.native_gguf_disk_tiering",
      "distributed_runtime.dense_tiering",
    ]));
  });

  it("rejects a subprocess that reports a passing shell but failed parity", () => {
    const stdout =
      INSTALLED_STAGE_CANARY_MARKER +
      JSON.stringify({
        ...validEvidence,
        parity: { ...validEvidence.parity, sequentialVsBatch: false },
      });
    expect(() =>
      runInstalledStageCanary(
        {
          pythonExecutable: "C:\\runtime\\python.exe",
          pythonSourceRoot: "C:\\resources\\python",
        },
        {
          existsSync: () => true,
          readFileSync: () => adapterRegistrySource,
          verifyNativePythonProductSource: acceptProductSource,
          spawnSync: () => ({
            status: 0,
            stdout,
            stderr: "",
          }),
        },
      ),
    ).toThrow("parity evidence is invalid");
  });

  it("rejects missing or duplicate result markers", () => {
    const invoke = (stdout: string) =>
      runInstalledStageCanary(
        {
          pythonExecutable: "C:\\runtime\\python.exe",
          pythonSourceRoot: "C:\\resources\\python",
        },
        {
          existsSync: () => true,
          readFileSync: () => adapterRegistrySource,
          verifyNativePythonProductSource: acceptProductSource,
          spawnSync: () => ({
            status: 0,
            stdout,
            stderr: "",
          }),
        },
      );
    expect(() => invoke("ordinary log")).toThrow("exactly one");
    const marker = INSTALLED_STAGE_CANARY_MARKER + JSON.stringify(validEvidence);
    expect(() => invoke(`${marker}\n${marker}`)).toThrow("exactly one");
  });
});
