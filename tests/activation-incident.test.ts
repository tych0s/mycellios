import { describe, expect, it } from "vitest";
import {
  activationFailureCanRetryAfterRuntimeChange,
  activationFailureIsTransient,
  classifyActivationIncident,
  formatExhaustedActivationFailure,
} from "../src/contracts/activation-incident.js";

describe("activation incident classification", () => {
  it("retries while two connected desktops finish executor verification", () => {
    const incident = classifyActivationIncident({
      message: "distributed_activation_requires_two_connected_shard_executors",
      retryCount: 0,
      nextRetryAt: Date.UTC(2026, 6, 29, 15, 0, 5),
      maximumAttempts: 5,
    });

    expect(incident).toMatchObject({
      code: "executor_pool_not_ready",
      scope: "network",
      repairState: "scheduled",
      automaticAction: "rebuild_route",
      retryable: true,
      automatic: true,
      attempt: 1,
    });
    expect(activationFailureIsTransient(
      "distributed_activation_requires_two_connected_shard_executors",
    )).toBe(true);
  });

  it("retries while fresh link evidence replaces degraded samples", () => {
    const incident = classifyActivationIncident({
      message:
        "no_feasible_automatic_2_stage_pipeline:route_availability_below_minimum",
      retryCount: 0,
      nextRetryAt: Date.UTC(2026, 6, 30, 15, 0, 5),
      maximumAttempts: 5,
    });

    expect(incident).toMatchObject({
      code: "route_evidence_not_ready",
      scope: "network",
      repairState: "scheduled",
      automaticAction: "rebuild_route",
      retryable: true,
      automatic: true,
      attempt: 1,
    });
    expect(activationFailureIsTransient(
      "no_feasible_automatic_2_stage_pipeline",
    )).toBe(true);
  });

  it("publishes a bounded autonomous node-recovery plan", () => {
    const incident = classifyActivationIncident({
      message: "distributed_worker_disconnected:wrk-68_320a",
      retryCount: 0,
      retryLaunching: false,
      nextRetryAt: Date.UTC(2026, 6, 25, 20, 0, 5),
      maximumAttempts: 5,
    });

    expect(incident).toEqual({
      schema: "mycellios-activation-incident/1",
      code: "node_disconnected",
      scope: "node",
      title: "A required node disconnected",
      summary:
        "The partial topology was not published because one of its physical executors left during startup.",
      remedy:
        "Mycellios waits for healthy capacity, discards the partial route and rebuilds it automatically.",
      steps: [
        "Retire any partial stage without publishing it.",
        "Recalculate placement from currently verified capacity.",
        "Run health checks and a real inference canary before publication.",
      ],
      retryable: true,
      automatic: true,
      repairState: "scheduled",
      automaticAction: "reconnect_node",
      attempt: 1,
      maximumAttempts: 5,
      nextRetryAt: "2026-07-25T20:00:05.000Z",
      nodeId: "wrk-68_320a",
      stageId: null,
      processExitCode: null,
    });
    expect(activationFailureIsTransient(
      "distributed_worker_disconnected:wrk-68_320a",
    )).toBe(true);
  });

  it("identifies a retrying accelerator OOM and its affected stage", () => {
    const incident = classifyActivationIncident({
      message:
        "launch_process_exited:stage-28a07ed:gpu_model_stage_unavailable_after_retries:cuda:out of memory:code=1",
      retryCount: 2,
      retryLaunching: true,
      nextRetryAt: Date.now(),
      maximumAttempts: 5,
    });

    expect(incident).toMatchObject({
      code: "out_of_memory",
      scope: "accelerator",
      repairState: "retrying",
      automaticAction: "reprofile_and_replan",
      retryable: true,
      automatic: true,
      attempt: 2,
      stageId: "stage-28a07ed",
      processExitCode: 1,
      nextRetryAt: null,
    });
  });

  it("stops a native access-violation loop and keeps only safe identifiers", () => {
    const incident = classifyActivationIncident({
      message:
        "launch_process_exited:stage-abc:C:\\Users\\ExampleUser\\secret.py:Bearer token:code=3221225477",
      maximumAttempts: 5,
    });

    expect(incident).toMatchObject({
      code: "native_memory_access",
      scope: "stage",
      repairState: "manual_required",
      automaticAction: "none",
      retryable: false,
      automatic: false,
      stageId: "stage-abc",
      processExitCode: 3_221_225_477,
    });
    expect(JSON.stringify(incident)).not.toContain("ExampleUser");
    expect(JSON.stringify(incident)).not.toContain("Bearer");
    expect(activationFailureIsTransient(
      "launch_process_exited:stage-abc:code=3221225477",
    )).toBe(false);
  });

  it("reports exhausted retries without offering another automatic action", () => {
    const incident = classifyActivationIncident({
      message:
        "automatic_activation_retries_exhausted:5:managed_launch_agent_is_unavailable:desktop-a",
      maximumAttempts: 5,
    });

    expect(incident).toMatchObject({
      code: "launch_agent_unavailable",
      repairState: "exhausted",
      retryable: false,
      automatic: false,
      automaticAction: "none",
      attempt: 5,
      maximumAttempts: 5,
      nodeId: "desktop-a",
    });
    expect(activationFailureIsTransient(
      "automatic_activation_retries_exhausted:5:managed_launch_agent_is_unavailable:desktop-a",
    )).toBe(false);
  });

  it("rearms an exhausted transient failure exactly once when the runtime changes", () => {
    const failure = formatExhaustedActivationFailure(
      5,
      "0.2.63",
      "managed_launch_agent_is_unavailable:desktop-a",
    );

    expect(failure).toBe(
      "automatic_activation_retries_exhausted:5:runtime=0.2.63:managed_launch_agent_is_unavailable:desktop-a",
    );
    expect(activationFailureCanRetryAfterRuntimeChange(failure, "0.2.63")).toBe(false);
    expect(activationFailureCanRetryAfterRuntimeChange(failure, "0.2.64")).toBe(true);
    expect(classifyActivationIncident({
      message: failure,
      maximumAttempts: 5,
    })).toMatchObject({
      code: "launch_agent_unavailable",
      repairState: "exhausted",
      attempt: 5,
      nodeId: "desktop-a",
    });
  });

  it("allows one migration retry for a legacy exhausted transient failure", () => {
    expect(activationFailureCanRetryAfterRuntimeChange(
      "automatic_activation_retries_exhausted:5:distributed_worker_disconnected:wrk-a",
      "0.2.64",
    )).toBe(true);
    expect(activationFailureCanRetryAfterRuntimeChange(
      "automatic_activation_retries_exhausted:5:unsupported_model_architecture:qwen-next",
      "0.2.64",
    )).toBe(false);
  });

  it("fails closed for unknown and incompatible failures", () => {
    expect(classifyActivationIncident({
      message: "something entirely new",
      maximumAttempts: 5,
    })).toMatchObject({
      code: "unknown",
      repairState: "manual_required",
      retryable: false,
      automatic: false,
    });

    expect(classifyActivationIncident({
      message: "unsupported_model_architecture:qwen-next",
      maximumAttempts: 5,
    })).toMatchObject({
      code: "model_incompatible",
      scope: "model",
      automaticAction: "none",
    });
  });
});
