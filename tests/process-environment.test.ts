import { describe, expect, it } from "vitest";
import {
  buildIsolatedProcessEnvironment,
  executorIsolationCapabilityFromPolicy,
  ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS,
  normalizeExecutorIsolationPolicy,
  validateExecutorIsolationPolicy,
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

  it("normalizes an honest versioned executor policy", () => {
    const policy = normalizeExecutorIsolationPolicy({
      maxOutputBytesPerStream: 128 * 1024,
      stopGraceMs: 12_000,
    });

    expect(policy).toMatchObject({
      schema: "gdlp-executor-isolation/4",
      environmentPolicy:
        "inherit-reviewed-system-keys-plus-trusted-overrides",
      executablePolicy: "exact-prepared-command",
      workspacePolicy: "private-temp-shared-runtime",
      processTreePolicy: "best-effort-process-tree",
      resourceLimitPolicy: "workspace-watchdog-only",
      maxOutputBytesPerStream: 128 * 1024,
      stopGraceMs: 12_000,
      maxWorkspaceBytes: 512 * 1024 * 1024,
      maxWorkspaceEntries: 10_000,
      workspaceCheckIntervalMs: 1_000,
    });
    expect(policy.inheritedEnvironmentKeys).toEqual(
      ISOLATED_PROCESS_INHERITED_ENVIRONMENT_KEYS,
    );
    expect(() => validateExecutorIsolationPolicy(policy)).not.toThrow();
  });

  it("rejects claims for controls the executor cannot enforce yet", () => {
    const policy = normalizeExecutorIsolationPolicy();
    const falseClaim = {
      ...policy,
      resourceLimitPolicy: "os-enforced",
    };

    expect(() => validateExecutorIsolationPolicy(falseClaim)).toThrow(
      "executor_isolation_resource_limit_policy_is_unsupported",
    );
  });

  it("rejects changes to the reviewed inherited environment", () => {
    const policy = normalizeExecutorIsolationPolicy();
    policy.inheritedEnvironmentKeys.push("MYCELLIOS_NETWORK_TOKEN");

    expect(() => validateExecutorIsolationPolicy(policy)).toThrow(
      "executor_isolation_environment_allowlist_mismatch",
    );
  });

  it("advertises the active controls without claiming an OS sandbox or hard quotas", () => {
    const capability = executorIsolationCapabilityFromPolicy(
      normalizeExecutorIsolationPolicy({
        maxWorkspaceBytes: 256 * 1024 * 1024,
        maxWorkspaceEntries: 2_500,
        workspaceCheckIntervalMs: 750,
      }),
    );

    expect(capability).toEqual({
      schema: "mycellios-executor-isolation-capability/1",
      launchPolicySchema: "gdlp-executor-isolation/4",
      environment: "filtered",
      workspace: "private-temp-watchdog",
      processTree: "best-effort",
      resourceLimits: "workspace-watchdog-only",
      osSandbox: "not-enforced",
      hardResourceQuotas: "not-enforced",
      killOnClose: "not-enforced",
      maxWorkspaceBytes: 256 * 1024 * 1024,
      maxWorkspaceEntries: 2_500,
      workspaceCheckIntervalMs: 750,
    });
  });
});
