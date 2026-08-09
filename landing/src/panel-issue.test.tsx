import { describe, expect, it } from "vitest";
import { panelIssueForErrors } from "./Panel";

describe("Panel issue classification", () => {
  it("does not label a local runtime failure as a coordinator outage", () => {
    expect(panelIssueForErrors(null, "Worker admission failed")).toEqual({
      source: "runtime",
      message: "Worker admission failed",
    });
  });

  it("keeps a real coordinator error authoritative", () => {
    expect(panelIssueForErrors("HTTP 503", "Worker admission failed")).toEqual({
      source: "coordinator",
      message: "HTTP 503",
    });
    expect(panelIssueForErrors(null, null)).toBeNull();
  });
});
