import { describe, expect, it } from "vitest";
import { boundaryViolation, componentFor } from "../src/program/component-boundaries.js";

describe("Control and Engine component boundaries", () => {
  it("classifies the deployables and shared contracts", () => {
    expect(componentFor("landing/src/Panel.tsx")).toBe("control");
    expect(componentFor("src/worker/agent.ts")).toBe("engine");
    expect(componentFor("src/contracts/worker-protocol.ts")).toBe("shared");
    expect(componentFor("src/adapters/mycellios-pipeline.ts")).toBe("engine");
    expect(componentFor("src/benchlab/run.ts")).toBe("tooling");
    expect(componentFor("src/economy/economic-ledger.ts")).toBe("engine");
    expect(componentFor("src/simulator/model.ts")).toBe("tooling");
    expect(componentFor("src/support/assistant.ts")).toBe("shared");
  });

  it("prevents browser code from importing Engine internals", () => {
    expect(boundaryViolation("landing/src/client.ts", "../../src/distribution/planners.js"))
      .toBe("web_must_use_control_api_not_engine_internals");
  });

  it("prevents Engine and shared contracts from depending on Control", () => {
    expect(boundaryViolation("src/worker/agent.ts", "../coordinator/server.js"))
      .toBe("engine_must_not_depend_on_control");
    expect(boundaryViolation("src/contracts/types.ts", "../coordinator/server.js"))
      .toBe("shared_must_not_depend_on_control");
  });
});
