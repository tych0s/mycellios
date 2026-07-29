import { describe, expect, it } from "vitest";
import {
  activationProgressEventsForModel,
  formatActivationTime,
  normalizeActivationProgressEvents,
} from "./Panel";

describe("activation progress presentation", () => {
  it("uses the newest terminal event and closes restored historical spinners", () => {
    const events = normalizeActivationProgressEvents([
      {
        phase: "publishing_model",
        message: "Canary passed. Publishing the model to the network.",
        at: "2026-07-24T08:18:45.469Z",
        state: "running",
      },
      {
        phase: "active",
        message: "Model active. Canary measured 1.68 tokens/s.",
        at: "2026-07-24T08:18:45.682Z",
        state: "completed",
      },
      {
        phase: "running_canary",
        message: "Running a real inference canary across the network.",
        at: "2026-07-24T08:18:38.660Z",
        state: "running",
      },
    ]);

    expect(events.map(({ phase, state }) => ({ phase, state }))).toEqual([
      { phase: "running_canary", state: "completed" },
      { phase: "publishing_model", state: "completed" },
      { phase: "active", state: "completed" },
    ]);
    expect(events.at(-1)?.phase).toBe("active");
  });

  it("makes the current model failure dominate a stale successful timeline", () => {
    const events = activationProgressEventsForModel({
      status: "failed",
      message: "Automatic activation failed: distributed_activation_requires_two_connected_shard_executors",
      activationRequestedAt: "2026-07-29T14:53:23.674Z",
      updatedAt: "2026-07-29T14:53:23.674Z",
      activationProgress: [
        {
          phase: "publishing_model",
          message: "Canary passed. Publishing the model to the network.",
          at: "2026-07-24T08:18:45.469Z",
          state: "running",
        },
        {
          phase: "active",
          message: "Model active. Canary measured 1.68 tokens/s.",
          at: "2026-07-24T08:18:45.682Z",
          state: "completed",
        },
      ],
    });

    expect(events.map(({ phase, state }) => ({ phase, state }))).toEqual([
      { phase: "publishing_model", state: "completed" },
      { phase: "active", state: "completed" },
      { phase: "failed", state: "failed" },
    ]);
    expect(events.at(-1)?.message).toBe(
      "Two verified shard executors are not ready yet. The model remains unpublished while Mycellios rebuilds the route.",
    );
  });

  it("shows the event date as well as its time", () => {
    const [date, time] = formatActivationTime("2026-07-24T08:18:45.682Z").split("\n");
    expect(date).toBeTruthy();
    expect(time).toMatch(/08:18:45/);
  });

  it("does not promise another retry after the bounded attempts are exhausted", () => {
    const events = activationProgressEventsForModel({
      status: "failed",
      message:
        "Automatic activation failed: automatic_activation_retries_exhausted:5:distributed_activation_requires_two_connected_shard_executors",
      activationRequestedAt: null,
      updatedAt: "2026-07-29T15:00:00.000Z",
      activationProgress: [],
    });

    expect(events.at(-1)?.message).toContain("paused after 5 attempts");
    expect(events.at(-1)?.message).toContain("run activation again");
    expect(events.at(-1)?.message).not.toContain("rebuilds the route");
  });
});
