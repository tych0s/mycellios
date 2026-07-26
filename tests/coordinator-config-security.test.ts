import { describe, expect, it } from "vitest";
import {
  assertCoordinatorNetworkSecurity,
  isCoordinatorLoopbackHost,
  loadCoordinatorConfig,
} from "../src/core/config.js";

describe("coordinator network configuration security", () => {
  it.each([
    "localhost",
    "127.0.0.1",
    "127.25.10.3",
    "::1",
    "[::1]",
    "0:0:0:0:0:0:0:1",
    "::ffff:127.0.0.1",
  ])("recognizes loopback host %s", (host) => {
    expect(isCoordinatorLoopbackHost(host)).toBe(true);
    expect(() => assertCoordinatorNetworkSecurity({ host })).not.toThrow();
  });

  it.each([
    "0.0.0.0",
    "::",
    "[::]",
    "192.168.1.20",
    "10.0.0.4",
    "127.example.com",
    "coordinator.internal",
  ])("rejects unauthenticated non-loopback host %s", (host) => {
    expect(isCoordinatorLoopbackHost(host)).toBe(false);
    expect(() => assertCoordinatorNetworkSecurity({ host })).toThrow(
      "MYCELLIOS_NETWORK_TOKEN is required when GPU_MESH_HOST is not loopback.",
    );
  });

  it("keeps the default local coordinator available without a network token", () => {
    const config = loadCoordinatorConfig({ GPU_MESH_DB: ":memory:" });
    expect(config.host).toBe("127.0.0.1");
    expect(config.networkToken).toBeUndefined();
  });

  it("allows a public bind only when the network token is present", () => {
    const config = loadCoordinatorConfig({
      GPU_MESH_HOST: " 0.0.0.0 ",
      GPU_MESH_DB: ":memory:",
      MYCELLIOS_NETWORK_TOKEN: " network-secret ",
    });
    expect(config.host).toBe("0.0.0.0");
    expect(config.networkToken).toBe("network-secret");
  });

  it("rejects a public bind during environment configuration loading", () => {
    expect(() => loadCoordinatorConfig({
      GPU_MESH_HOST: "0.0.0.0",
      GPU_MESH_DB: ":memory:",
    })).toThrow(
      "MYCELLIOS_NETWORK_TOKEN is required when GPU_MESH_HOST is not loopback.",
    );
  });
});
