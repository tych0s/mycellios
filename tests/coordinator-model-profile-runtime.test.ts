import { describe, expect, it } from "vitest";
import { parseAutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import { configureCoordinatorModelProfileRuntime } from "../src/coordinator/model-profile-runtime.js";

describe("coordinator model profile runtime", () => {
  it("replaces durable paths from an old host with the container runtime", () => {
    const configured = configureCoordinatorModelProfileRuntime(baseConfig(), {
      runtimeRoot: "/app",
      platform: "linux",
      exists: (path) => path === "/opt/python/bin/python",
      environment: {},
    });

    expect(configured.runtime.pythonExecutable).toBe("/opt/python/bin/python");
    expect(configured.runtime.pythonPath).toBe("/app/python");
    expect(configured.runtime.hfHome).toBe("/var/lib/mycellios/hf-cache");
  });

  it("uses an explicit provisioned development runtime", () => {
    const configured = configureCoordinatorModelProfileRuntime(baseConfig(), {
      runtimeRoot: "C:\\workspace",
      platform: "win32",
      exists: (path) => path === "C:\\runtime\\python.exe",
      environment: {
        MYCELLIOS_MODEL_PROFILE_PYTHON: "C:\\runtime\\python.exe",
      },
    });

    expect(configured.runtime.pythonExecutable).toBe("C:\\runtime\\python.exe");
  });

  it("fails at startup when the selected runtime is absent", () => {
    expect(() => configureCoordinatorModelProfileRuntime(baseConfig(), {
      runtimeRoot: "/app",
      platform: "linux",
      exists: () => false,
      environment: {},
    })).toThrow("coordinator_model_profile_runtime_is_not_provisioned");
  });

  it("fails closed for a missing explicit runtime", () => {
    expect(() => configureCoordinatorModelProfileRuntime(baseConfig(), {
      runtimeRoot: "/app",
      platform: "linux",
      exists: () => false,
      environment: {
        MYCELLIOS_MODEL_PROFILE_PYTHON: "/missing/python",
      },
    })).toThrow("coordinator_model_profile_python_not_found:/missing/python");
  });
});

function baseConfig() {
  return parseAutoDistributionConfig({
    schema: "gdlp-auto-distribute/1",
    model: {
      source: "hmellor/tiny-random-LlamaForCausalLM",
      revision: null,
      publicName: "tiny-random",
    },
    nodes: [
      node("node-a", 9_801),
      node("node-b", 9_802),
    ],
    runtime: {
      pythonExecutable: "/home/old-host/distribution-venv/bin/python",
      pythonPath: "/home/old-host/mycellios/python",
      hfHome: "/var/lib/mycellios/hf-cache",
      apiEndpoint: { host: "127.0.0.1", port: 8_892 },
      returnEndpoint: { host: "127.0.0.1", port: 30_092 },
    },
    coordinator: {
      url: "http://127.0.0.1:8787",
      region: "test",
      maxConcurrency: 1,
    },
  });
}

function node(id: string, port: number) {
  return {
    id,
    region: "test",
    endpoint: { host: "127.0.0.1", port },
    memoryMiB: 4_096,
    reserveMiB: 256,
    agent: { kind: "managed" as const },
  };
}
