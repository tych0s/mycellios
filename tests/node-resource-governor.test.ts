import { describe, expect, it } from "vitest";
import { breachedLimit, NodeResourceGovernor, type NodeResourceObservation } from "../src/node/resource-governor.js";

const limits = {
  maxConcurrency: 2,
  maxCpuPercent: 80,
  maxRamMiB: 8_192,
  maxVramMiB: 6_144,
  maxDiskMiB: 32_768,
  maxTemperatureC: 85,
};

describe("NodeResourceGovernor", () => {
  it("classifies every enforced limit deterministically with thermal priority", () => {
    expect(breachedLimit(limits, observation())).toBeNull();
    expect(breachedLimit(limits, observation({ cpuPercent: 81 }))).toBe("node_cpu_limit_exceeded");
    expect(breachedLimit(limits, observation({ ramMiB: 8_193 }))).toBe("node_ram_limit_exceeded");
    expect(breachedLimit(limits, observation({ diskMiB: 32_769 }))).toBe("node_disk_limit_exceeded");
    expect(breachedLimit(limits, observation({ temperatureC: 86 }))).toBe("node_temperature_limit_exceeded");
    expect(breachedLimit(limits, observation({ cpuPercent: 99, temperatureC: 90 })))
      .toBe("node_temperature_limit_exceeded");
  });

  it("invokes bounded teardown once when a live observation breaches policy", async () => {
    const breaches: string[] = [];
    const governor = new NodeResourceGovernor(
      limits,
      async () => observation({ ramMiB: 9_000 }),
      async (code) => { breaches.push(code); },
      100,
    );
    expect(await governor.evaluate()).toBe("node_ram_limit_exceeded");
    expect(breaches).toEqual(["node_ram_limit_exceeded"]);
    governor.stop();
  });

  it("fails closed when resource observation itself becomes unavailable", async () => {
    const breaches: string[] = [];
    const governor = new NodeResourceGovernor(
      limits,
      async () => { throw new Error("probe unavailable"); },
      async (code) => { breaches.push(code); },
      100,
    );
    governor.start();
    await new Promise((resolve) => setTimeout(resolve, 140));
    expect(breaches).toEqual(["node_resource_probe_failed"]);
    governor.stop();
  });
});

function observation(overrides: Partial<NodeResourceObservation> = {}): NodeResourceObservation {
  return {
    cpuPercent: 20,
    ramMiB: 1_024,
    diskMiB: 5_000,
    temperatureC: 60,
    ...overrides,
  };
}
