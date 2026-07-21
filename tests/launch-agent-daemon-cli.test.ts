import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  loadAllowedPythonLaunchDescription,
  parseLaunchAgentDaemonArguments,
  readLaunchAgentDaemonAuthToken,
} from "../src/distribution/launch-agent-daemon-cli.js";
import { compilePythonLaunchDescription } from "../src/distribution/python-launcher.js";
import {
  buildRuntimePipelineManifest,
  type RuntimePlanRequest,
} from "../src/distribution/runtime-manifest.js";
import type {
  DistributedModelProfile,
  DistributionPlan,
  DistributionWorkload,
} from "../src/distribution/types.js";

describe("launch-agent daemon CLI security configuration", () => {
  it("defaults to loopback and reads the bearer credential only from the default env", () => {
    const options = parseLaunchAgentDaemonArguments(["--node-id", "node-a"]);

    expect(options.host).toBe("127.0.0.1");
    expect(options.port).toBe(9_750);
    expect(options.authTokenEnv).toBe("GDLP_LAUNCH_AGENT_TOKEN");
    expect(options.physicalProbePython).toBe("python");
    expect(options.physicalProbeTimeoutMs).toBe(30_000);
    expect(
      readLaunchAgentDaemonAuthToken(options.authTokenEnv, {
        GDLP_LAUNCH_AGENT_TOKEN: "daemon-secret-01234567890123456789",
      }),
    ).toBe("daemon-secret-01234567890123456789");
  });

  it("configures only a fixed probe executable and timeout", () => {
    const options = parseLaunchAgentDaemonArguments([
      "--node-id",
      "node-a",
      "--physical-probe-python",
      "runtime/distribution-venv/bin/python",
      "--physical-probe-timeout-ms",
      "45000",
    ]);
    expect(options.physicalProbePython).toBe(
      "runtime/distribution-venv/bin/python",
    );
    expect(options.physicalProbeTimeoutMs).toBe(45_000);
    expect(() =>
      parseLaunchAgentDaemonArguments([
        "--node-id",
        "node-a",
        "--physical-probe-command",
        "arbitrary argv is forbidden",
      ]),
    ).toThrow("launch_agent_daemon_argument_is_invalid:--physical-probe-command");
  });

  it("allows selecting the credential env name without accepting a secret argv", () => {
    const options = parseLaunchAgentDaemonArguments([
      "--node-id",
      "node-a",
      "--auth-token-env",
      "PRIVATE_AGENT_TOKEN",
    ]);
    expect(options.authTokenEnv).toBe("PRIVATE_AGENT_TOKEN");
    expect(
      readLaunchAgentDaemonAuthToken(options.authTokenEnv, {
        PRIVATE_AGENT_TOKEN: "from-environment",
      }),
    ).toBe("from-environment");
    expect(readLaunchAgentDaemonAuthToken(options.authTokenEnv, {})).toBeUndefined();
    expect(
      readLaunchAgentDaemonAuthToken(options.authTokenEnv, {
        PRIVATE_AGENT_TOKEN: "",
      }),
    ).toBeUndefined();

    const secret = "must-not-appear-in-errors";
    let error: unknown;
    try {
      parseLaunchAgentDaemonArguments([
        "--node-id",
        "node-a",
        "--auth-token",
        secret,
      ]);
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(Error);
    expect((error as Error).message).not.toContain(secret);
  });

  it("rejects an invalid environment variable name without reading it", () => {
    expect(() =>
      parseLaunchAgentDaemonArguments([
        "--node-id",
        "node-a",
        "--auth-token-env",
        "NOT-AN-ENV-NAME",
      ]),
    ).toThrow("launch_agent_daemon_auth_token_env_is_invalid");
  });

  it("parses only a file path for the sealed launch and remains import-safe", () => {
    const options = parseLaunchAgentDaemonArguments([
      "--node-id",
      "node-a",
      "--allow-launch-file",
      "runtime/allowed-launch.json",
    ]);
    expect(options.allowLaunchFile).toBe("runtime/allowed-launch.json");
    expect(typeof loadAllowedPythonLaunchDescription).toBe("function");

    expect(() =>
      parseLaunchAgentDaemonArguments([
        "--node-id",
        "node-a",
        "--allow-launch-json",
        "inline-is-forbidden",
      ]),
    ).toThrow("launch_agent_daemon_argument_is_invalid:--allow-launch-json");
  });

  it("loads and closed-validates one gdlp-python-launch/2 allowlist file", async () => {
    const directory = await mkdtemp(join(tmpdir(), "gdlp-launch-allowlist-"));
    const path = join(directory, "allowed-launch.json");
    try {
      const description = daemonLaunchDescription();
      await writeFile(path, JSON.stringify(description), "utf8");
      await expect(loadAllowedPythonLaunchDescription(path)).resolves.toEqual(description);

      await writeFile(
        path,
        JSON.stringify({ ...description, launchId: "mutated-launch-id" }),
        "utf8",
      );
      await expect(loadAllowedPythonLaunchDescription(path)).rejects.toThrow(
        "python_launch_description_mismatch",
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });
});

function daemonLaunchDescription() {
  const mib = 1024 * 1024;
  const model: DistributedModelProfile = {
    id: "daemon-allowlist-model",
    layers: Array.from({ length: 2 }, (_, index) => ({
      index,
      weightBytes: mib,
      activationElements: 64,
      kvBytesPerToken: 16,
      decodeMsAtUnit: 1,
      prefillMsPerTokenAtUnit: 0.1,
    })),
    embeddingBytes: mib,
    lmHeadBytes: mib,
    runtimeOverheadBytesPerStage: mib,
    embeddingDecodeMsAtUnit: 0.1,
    lmHeadDecodeMsAtUnit: 0.1,
    embeddingPrefillMsPerTokenAtUnit: 0.01,
    lmHeadPrefillMsPerTokenAtUnit: 0.01,
  };
  const workload: DistributionWorkload = {
    promptTokens: 4,
    outputTokens: 2,
    contextTokens: 8,
    concurrentSequences: 1,
    maxStages: 2,
    maxQualityLoss: 0,
    minRouteAvailability: 0.8,
    batchWindowMs: 0,
    p95: false,
  };
  const plan: DistributionPlan = {
    algorithm: "daemon-allowlist-fixture",
    codec: "fp16",
    microBatchSize: 1,
    prefillChunkTokens: 4,
    stages: [
      { nodeId: "node-a", layerStart: 0, layerEnd: 1 },
      { nodeId: "node-b", layerStart: 1, layerEnd: 2 },
    ],
  };
  const nodes = ["node-a", "node-b"].map((id, index) => ({
    id,
    region: "test",
    memoryBytes: 16 * mib,
    reserveBytes: mib,
    decodeScale: 1,
    prefillScale: 1,
    codecScale: 1,
    batchGain: 0,
    maxBatchSpeedup: 1,
    powerWatts: 50,
    availability: 0.999,
    endpoint: { host: "127.0.0.1", port: 23_000 + index },
  }));
  const request: RuntimePlanRequest = {
    model,
    modelRevision: "sha256:daemon-allowlist-r1",
    topology: {
      nodes,
      links: [
        {
          from: "node-a",
          to: "node-b",
          oneWayLatencyMs: 1,
          jitterP95Ms: 0,
          bandwidthMbps: 1_000,
          lossRate: 0,
          availability: 0.999,
        },
        {
          from: "node-b",
          to: "node-a",
          oneWayLatencyMs: 1,
          jitterP95Ms: 0,
          bandwidthMbps: 1_000,
          lossRate: 0,
          availability: 0.999,
        },
      ],
    },
    workload,
    phasePlans: { prefill: plan, decode: plan },
  };
  return compilePythonLaunchDescription(buildRuntimePipelineManifest(request), {
    apiEndpoint: { host: "127.0.0.1", port: 8_081 },
    returnEndpoint: { host: "127.0.0.1", port: 30_000 },
    returnBindHost: "127.0.0.1",
  });
}
