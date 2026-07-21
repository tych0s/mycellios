import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  PYTHON_LAUNCH_REQUEST_SCHEMA,
  compilePythonLaunchRequest,
  executePythonLauncherCli,
  parsePythonLaunchRequest,
  parsePythonLauncherCliArguments,
} from "../src/distribution/python-launcher-cli.js";
import {
  validatePythonLaunchDescription,
  type PythonLaunchCompilerOptions,
} from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DirectedLinkProfile,
  DistributedModelProfile,
  DistributionPlan,
} from "../src/distribution/types.js";

const MIB = 1024 * 1024;

describe("python launcher CLI", () => {
  it("parses the required input and optional output flags in either form", () => {
    expect(
      parsePythonLauncherCliArguments([
        "--out=artifacts/launch.json",
        "--input",
        "request.json",
      ]),
    ).toEqual({
      inputPath: "request.json",
      outputPath: "artifacts/launch.json",
    });
    expect(parsePythonLauncherCliArguments(["--input=request.json"])).toEqual({
      inputPath: "request.json",
      outputPath: null,
    });
  });

  it("rejects missing, duplicate, unknown and positional arguments", () => {
    expect(() => parsePythonLauncherCliArguments([])).toThrow(
      "python_launcher_cli_input_is_required",
    );
    expect(() =>
      parsePythonLauncherCliArguments(["--input", "a.json", "--input=b.json"]),
    ).toThrow("python_launcher_cli_duplicate_input");
    expect(() =>
      parsePythonLauncherCliArguments([
        "--input",
        "a.json",
        "--out",
        "a.out.json",
        "--out=b.out.json",
      ]),
    ).toThrow("python_launcher_cli_duplicate_output");
    expect(() => parsePythonLauncherCliArguments(["--input", "--out", "x"])).toThrow(
      "python_launcher_cli_option_requires_value:--input",
    );
    expect(() => parsePythonLauncherCliArguments(["--input="])).toThrow(
      "python_launcher_cli_option_requires_value:--input",
    );
    expect(() => parsePythonLauncherCliArguments(["--input", "a", "--run"])).toThrow(
      "python_launcher_cli_unknown_option:--run",
    );
    expect(() => parsePythonLauncherCliArguments(["request.json"])).toThrow(
      "python_launcher_cli_positional_arguments_are_not_supported",
    );
  });

  it("accepts only the exact versioned request envelope", () => {
    const value = launchRequest();
    expect(parsePythonLaunchRequest(value)).toEqual(value);

    expect(() => parsePythonLaunchRequest(null)).toThrow(
      "python_launch_request_must_be_an_object",
    );
    expect(() => parsePythonLaunchRequest({ ...value, extra: true })).toThrow(
      "python_launch_request_must_have_exact_keys",
    );
    const { options: _options, ...missing } = value;
    expect(() => parsePythonLaunchRequest(missing)).toThrow(
      "python_launch_request_must_have_exact_keys",
    );
    expect(() => parsePythonLaunchRequest({ ...value, schema: "future/1" })).toThrow(
      "unsupported_python_launch_request_schema",
    );
    expect(() => parsePythonLaunchRequest({ ...value, manifest: [] })).toThrow(
      "python_launch_request_manifest_must_be_an_object",
    );
    expect(() => parsePythonLaunchRequest({ ...value, options: null })).toThrow(
      "python_launch_request_options_must_be_an_object",
    );
  });

  it("compiles the same validated request deterministically", () => {
    const value = launchRequest();
    const first = compilePythonLaunchRequest(value);
    const second = compilePythonLaunchRequest(structuredClone(value));

    expect(second).toEqual(first);
    expect(first.schema).toBe("gdlp-python-launch/2");
    expect(first.sourceManifest).toEqual(value.manifest);
    expect(first.launchOrder.map((entry) => entry.kind)).toEqual([
      "remote-stage",
      "remote-stage",
      "root-engine",
    ]);
    validatePythonLaunchDescription(first);
  });

  it("writes exactly one JSON document to injected stdout when --out is absent", async () => {
    const value = launchRequest();
    const virtualCwd = resolve("virtual-python-launch-cli");
    let observedInput = "";
    let stdout = "";
    let outputWrites = 0;

    const result = await executePythonLauncherCli(["--input", "request.json"], {
      cwd: virtualCwd,
      readText: async (path) => {
        observedInput = path;
        return JSON.stringify(value);
      },
      writeTextAtomic: async () => {
        outputWrites += 1;
      },
      writeStdout: (content) => {
        stdout += content;
      },
    });

    expect(observedInput).toBe(resolve(virtualCwd, "request.json"));
    expect(outputWrites).toBe(0);
    expect(stdout.endsWith("\n")).toBe(true);
    expect(JSON.parse(stdout)).toEqual(result);
  });

  it("resolves --out and uses the injected atomic writer without touching stdout", async () => {
    const value = launchRequest();
    const virtualCwd = resolve("virtual-python-launch-cli");
    const writes: Array<{ path: string; content: string }> = [];

    const result = await executePythonLauncherCli(
      ["--input=request.json", "--out", "nested/launch.json"],
      {
        cwd: virtualCwd,
        readText: async () => JSON.stringify(value),
        writeTextAtomic: async (path, content) => {
          writes.push({ path, content });
        },
        writeStdout: () => {
          throw new Error("stdout_must_not_be_written");
        },
      },
    );

    expect(writes).toHaveLength(1);
    expect(writes[0]!.path).toBe(resolve(virtualCwd, "nested/launch.json"));
    expect(writes[0]!.content.endsWith("\n")).toBe(true);
    expect(JSON.parse(writes[0]!.content)).toEqual(result);
  });

  it("never overwrites the launch request with its compiled output", async () => {
    let reads = 0;
    await expect(
      executePythonLauncherCli(
        ["--input", "request.json", "--out", join(".", "request.json")],
        {
          cwd: resolve("virtual-python-launch-cli"),
          readText: async () => {
            reads += 1;
            return JSON.stringify(launchRequest());
          },
        },
      ),
    ).rejects.toThrow("python_launcher_cli_output_must_not_overwrite_input");
    expect(reads).toBe(0);
  });

  it("publishes a real output atomically and leaves no temporary sibling", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gdlp-python-launch-cli-"));
    try {
      const inputPath = join(directory, "request.json");
      const outputPath = join(directory, "nested", "launch.json");
      await writeFile(inputPath, JSON.stringify(launchRequest()), "utf8");

      const result = await executePythonLauncherCli(
        ["--input", inputPath, "--out", outputPath],
        {
          writeStdout: () => {
            throw new Error("stdout_must_not_be_written");
          },
        },
      );

      expect(JSON.parse(await readFile(outputPath, "utf8"))).toEqual(result);
      expect(await readdir(join(directory, "nested"))).toEqual(["launch.json"]);
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("fails closed before any output for invalid JSON, manifests or options", async () => {
    let writes = 0;
    await expect(
      executePythonLauncherCli(["--input", "request.json"], {
        readText: async () => "{not-json",
        writeStdout: () => {
          writes += 1;
        },
      }),
    ).rejects.toThrow("python_launcher_cli_input_is_not_json");

    expect(() =>
      compilePythonLaunchRequest({
        schema: PYTHON_LAUNCH_REQUEST_SCHEMA,
        manifest: {},
        options: launchOptions(),
      }),
    ).toThrow();
    expect(() =>
      compilePythonLaunchRequest({
        ...launchRequest(),
        options: launchOptions({ returnEndpoint: { host: "root.internal", port: 8_081 } }),
      }),
    ).toThrow("python_api_and_return_ports_conflict");
    expect(writes).toBe(0);
  });
});

function launchRequest() {
  return {
    schema: PYTHON_LAUNCH_REQUEST_SCHEMA,
    manifest: buildRuntimePipelineManifest(runtimeRequest()),
    options: launchOptions(),
  };
}

function launchOptions(
  overrides: Partial<PythonLaunchCompilerOptions> = {},
): PythonLaunchCompilerOptions {
  return {
    apiEndpoint: { host: "0.0.0.0", port: 8_081 },
    returnEndpoint: { host: "root.internal", port: 30_000 },
    returnBindHost: "0.0.0.0",
    pythonExecutable: "python",
    threadsPerStage: 2,
    connectTimeoutSeconds: 45,
    ...overrides,
  };
}

function runtimeRequest(): RuntimePlanRequest {
  const nodes: RuntimePlanRequest["topology"]["nodes"] = Array.from(
    { length: 3 },
    (_, index) => ({
      id: `node-${index}`,
      region: "test-lan",
      memoryBytes: 128 * MIB,
      reserveBytes: 8 * MIB,
      decodeScale: 1,
      prefillScale: 1,
      codecScale: 1,
      batchGain: 0.1,
      maxBatchSpeedup: 1.2,
      powerWatts: 75,
      availability: 0.999,
      endpoint: { host: `stage-${index}.internal`, port: 22_000 + index },
      backend: {
        engine: "python-transformers",
        version: "1",
        modelFormats: ["safetensors"],
        executionModes: ["layer-range"],
      },
      capabilities: {
        deviceKinds: ["gpu"],
        computeApis: ["cuda"],
        weightDtypes: ["fp16"],
        activationCodecs: ["fp16", "int8"],
        features: ["layer-range", "kv-reuse", "kv-transfer"],
      },
    }),
  );
  const links: DirectedLinkProfile[] = [];
  for (const from of nodes) {
    for (const to of nodes) {
      if (from.id === to.id) continue;
      links.push({
        from: from.id,
        to: to.id,
        oneWayLatencyMs: 1,
        jitterP95Ms: 0.1,
        bandwidthMbps: 1_000,
        lossRate: 0,
        availability: 0.999,
      });
    }
  }
  const plan: DistributionPlan = {
    algorithm: "python-launch-cli-fixture",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 8,
    stages: nodes.map((node, index) => ({
      nodeId: node.id,
      layerStart: index * 2,
      layerEnd: (index + 1) * 2,
    })),
  };
  return {
    model: modelProfile(),
    modelRevision: "sha256:python-launch-cli-model-r1",
    tokenizerId: "python-launch-cli-tokenizer-r1",
    topology: { nodes, links },
    workload: {
      promptTokens: 16,
      outputTokens: 8,
      contextTokens: 32,
      concurrentSequences: 1,
      maxStages: 3,
      maxQualityLoss: 1,
      minRouteAvailability: 0.9,
      batchWindowMs: 1,
      p95: false,
    },
    phasePlans: { prefill: plan, decode: plan },
  };
}

function modelProfile(): DistributedModelProfile {
  return {
    id: "python-launch-cli-model",
    layers: Array.from({ length: 6 }, (_, index) => ({
      index,
      weightBytes: 24 * MIB,
      activationElements: 512,
      kvBytesPerToken: 128,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: MIB,
    lmHeadBytes: MIB,
    runtimeOverheadBytesPerStage: 4 * MIB,
    embeddingDecodeMsAtUnit: 0.2,
    lmHeadDecodeMsAtUnit: 0.2,
    embeddingPrefillMsPerTokenAtUnit: 0.05,
    lmHeadPrefillMsPerTokenAtUnit: 0.05,
  };
}
