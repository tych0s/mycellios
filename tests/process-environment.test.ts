import { describe, expect, it } from "vitest";
import {
  buildIsolatedProcessEnvironment,
  ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS,
} from "../src/distribution/process-environment.js";

describe("isolated process environment", () => {
  it("inherits only reviewed system keys and explicit overrides", () => {
    const environment = buildIsolatedProcessEnvironment({
      source: {
        PATH: "safe-path",
        TEMP: "safe-temp",
        MYCELLIOS_NETWORK_TOKEN: "network-secret",
        HF_TOKEN: "model-secret",
        AWS_SECRET_ACCESS_KEY: "cloud-secret",
        HTTPS_PROXY: "https://user:password@example.invalid",
      },
      overrides: {
        PYTHONPATH: "runtime-python",
        HF_HOME: "model-cache",
      },
    });

    expect(environment).toEqual({
      PATH: "safe-path",
      TEMP: "safe-temp",
      PYTHONPATH: "runtime-python",
      HF_HOME: "model-cache",
    });
  });

  it("lets an explicit policy remove or grant one value", () => {
    const environment = buildIsolatedProcessEnvironment({
      source: {
        PATH: "parent-path",
        HOME: "parent-home",
        PRIVATE_VALUE: "not-inherited",
      },
      overrides: {
        PATH: undefined,
        PRIVATE_VALUE: "reviewed-value",
      },
    });

    expect(environment).toEqual({
      HOME: "parent-home",
      PRIVATE_VALUE: "reviewed-value",
    });
  });

  it("does not add credential-shaped names to the default allowlist", () => {
    expect(ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS).not.toContain(
      "MYCELLIOS_NETWORK_TOKEN",
    );
    expect(ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS).not.toContain("HF_TOKEN");
    expect(ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS).not.toContain(
      "AWS_SECRET_ACCESS_KEY",
    );
    expect(ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS).not.toContain(
      "HTTPS_PROXY",
    );
  });
});
