import { describe, expect, it } from "vitest";
import { selectPreferredLanAddress } from "../src/desktop/network.js";

describe("desktop network selection", () => {
  it("prefers a physical LAN adapter over an earlier VPN adapter", () => {
    expect(selectPreferredLanAddress({
      NordLynx: [address("10.5.0.2")],
      "Wi-Fi": [address("192.168.1.79")],
    })).toBe("192.168.1.79");
  });

  it("falls back to loopback when no usable IPv4 address exists", () => {
    expect(selectPreferredLanAddress({ Loopback: [address("127.0.0.1", true)] }))
      .toBe("127.0.0.1");
  });
});

function address(value: string, internal = false) {
  return {
    address: value,
    netmask: "255.255.255.0",
    family: "IPv4" as const,
    mac: "00:00:00:00:00:00",
    internal,
    cidr: `${value}/24`,
  };
}
