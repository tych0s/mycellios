import { describe, expect, it } from "vitest";
import {
  validatePhysicalProbe,
  type PhysicalProbeV1,
} from "../src/distribution/physical-probe.js";

describe("physical GPU probe contract", () => {
  it("accepts a closed, nonce-bound physical probe", () => {
    const value = fixture();
    expect(() => validatePhysicalProbe(value, value.nonce)).not.toThrow();
  });

  it("rejects replay, duplicate GPUs, non-finite memory and extra fields", () => {
    const replay = fixture();
    expect(() => validatePhysicalProbe(replay, "different-campaign-0002")).toThrow(
      "physical_probe_nonce_does_not_match",
    );

    const duplicate = fixture();
    duplicate.devices.push(structuredClone(duplicate.devices[0]!));
    expect(() => validatePhysicalProbe(duplicate)).toThrow(
      "physical_probe_devices_are_not_unique",
    );

    const infinite = fixture();
    infinite.devices[0]!.totalMemoryBytes = Number.POSITIVE_INFINITY;
    expect(() => validatePhysicalProbe(infinite)).toThrow(
      "physical_probe_device_memory_is_invalid",
    );

    const extra = fixture() as PhysicalProbeV1 & { hostname?: string };
    extra.hostname = "private-hostname";
    expect(() => validatePhysicalProbe(extra)).toThrow("physical_probe_keys_are_invalid");
  });
});

function fixture(): PhysicalProbeV1 {
  return {
    schema: "gdlp-physical-probe/1",
    nonce: "physical-campaign-0001",
    host: {
      fingerprintSha256: `sha256:${"1".repeat(64)}`,
      fingerprintSource: "test",
      platform: "linux",
      architecture: "x86_64",
      kernelRelease: "6.8",
      pythonVersion: "3.12.0",
    },
    runtime: {
      torchVersion: "2.13.0",
      cudaVersion: "13.0",
      rocmVersion: null,
      cudaApiAvailable: true,
      distributedAvailable: true,
      ncclAvailable: true,
      ncclVersion: "2.25.1",
    },
    devices: [
      {
        index: 0,
        name: "Test GPU",
        totalMemoryBytes: 4 * 1024 ** 3,
        freeMemoryBytes: 3 * 1024 ** 3,
        runtimeTotalMemoryBytes: 4 * 1024 ** 3,
        capability: [8, 6],
        uuidSha256: `sha256:${"2".repeat(64)}`,
        fingerprintSha256: `sha256:${"3".repeat(64)}`,
      },
    ],
  };
}
