import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CompositeTcpPortMapper,
} from "../src/worker/native-port-mapper.js";
import {
  decodeNatPmpExternalAddress,
  decodeNatPmpMapResponse,
  decodePcpMapResponse,
  PcpNatPmpPortMapper,
  type GatewayUdpExchange,
  type NativeIpv4Gateway,
} from "../src/worker/pcp-nat-pmp-port-mapper.js";
import type {
  TcpPortMapper,
  TcpPortMapping,
} from "../src/worker/upnp-port-mapper.js";

const GATEWAY: NativeIpv4Gateway = {
  gatewayAddress: "192.168.1.1",
  localAddress: "192.168.1.20",
  interfaceName: "test-lan",
  metric: 10,
};
const NONCE = Buffer.from("000102030405060708090a0b", "hex");

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("native PCP and NAT-PMP port mapping", () => {
  it("encodes, validates, refreshes and deletes one nonce-bound PCP TCP lease", async () => {
    vi.useFakeTimers();
    const requests: Buffer[] = [];
    const exchange: GatewayUdpExchange = vi.fn(async ({ request }) => {
      requests.push(Buffer.from(request));
      expect(request[0]).toBe(2);
      expect(request[1]).toBe(1);
      expect(request.subarray(24, 36)).toEqual(NONCE);
      const lifetime = request.readUInt32BE(4);
      return pcpResponse(request, lifetime, "8.8.8.8", 45_678);
    });
    const mapper = mapperWith(exchange);
    const mapping = await mapper.mapTcpPort(input());
    expect(mapping).toMatchObject({
      externalHost: "8.8.8.8",
      externalPort: 45_678,
    });
    expect(requests).toHaveLength(1);

    await vi.advanceTimersByTimeAsync(30_000);
    expect(requests).toHaveLength(2);
    expect(requests[1]?.readUInt32BE(4)).toBe(60);

    await mapping?.close();
    await mapping?.close();
    expect(requests).toHaveLength(3);
    expect(requests[2]?.readUInt32BE(4)).toBe(0);
  });

  it("falls back from PCP to NAT-PMP, observes the public address and removes the lease", async () => {
    const requests: Buffer[] = [];
    const exchange: GatewayUdpExchange = vi.fn(async ({ request }) => {
      requests.push(Buffer.from(request));
      if (request[0] === 2) throw new Error("pcp_not_supported");
      if (request.byteLength === 2) return natPmpAddressResponse("1.1.1.1");
      return natPmpMapResponse(
        request.readUInt16BE(4),
        40_001,
        request.readUInt32BE(8),
      );
    });
    const mapping = await mapperWith(exchange).mapTcpPort(input());
    expect(mapping).toMatchObject({
      externalHost: "1.1.1.1",
      externalPort: 40_001,
    });
    expect(requests.map((request) => `${request[0]}:${request[1]}`))
      .toEqual(["2:1", "0:0", "0:2"]);
    await mapping?.close();
    expect(requests.at(-1)?.readUInt32BE(8)).toBe(0);
  });

  it("returns no candidate for malformed, private or mismatched protocol replies", async () => {
    const privateAddressExchange: GatewayUdpExchange = vi.fn(async ({ request }) => {
      if (request[0] === 2) return Buffer.alloc(60);
      if (request.byteLength === 2) return natPmpAddressResponse("100.64.1.2");
      throw new Error("unexpected_map");
    });
    await expect(mapperWith(privateAddressExchange).mapTcpPort(input()))
      .resolves.toBeNull();

    expect(() => decodePcpMapResponse(Buffer.alloc(60), {
      nonce: NONCE,
      internalPort: 34_567,
      deletion: false,
    })).toThrow("pcp_map_response_is_invalid");
    expect(() => decodeNatPmpExternalAddress(natPmpAddressResponse("192.168.1.5")))
      .toThrow("nat_pmp_external_address_is_not_public");
    expect(() => decodeNatPmpMapResponse(
      natPmpMapResponse(34_568, 40_001, 60),
      { internalPort: 34_567, deletion: false },
    )).toThrow("nat_pmp_map_response_is_invalid");
  });

  it("invalidates an advertised lease when bounded refresh retries reach expiry", async () => {
    vi.useFakeTimers();
    let exchanges = 0;
    const exchange: GatewayUdpExchange = vi.fn(async ({ request }) => {
      exchanges += 1;
      if (exchanges > 1) throw new Error("gateway_unreachable");
      return pcpResponse(request, 60, "8.8.8.8", 45_678);
    });
    const mapping = await mapperWith(exchange).mapTcpPort(input());
    const invalidated = vi.fn();
    mapping?.onInvalidated?.(invalidated);
    await vi.advanceTimersByTimeAsync(61_000);
    expect(invalidated).toHaveBeenCalledTimes(1);
  });

  it("invalidates instead of trusting a lease across a gateway epoch restart", async () => {
    vi.useFakeTimers();
    let exchanges = 0;
    const exchange: GatewayUdpExchange = vi.fn(async ({ request }) => {
      exchanges += 1;
      return pcpResponse(
        request,
        request.readUInt32BE(4),
        "8.8.8.8",
        45_678,
        exchanges === 1 ? 100 : 1,
      );
    });
    const mapping = await mapperWith(exchange).mapTcpPort(input());
    const invalidated = vi.fn();
    mapping?.onInvalidated?.(invalidated);
    await vi.advanceTimersByTimeAsync(30_000);
    expect(invalidated).toHaveBeenCalledTimes(1);
    expect(exchanges).toBe(3);
  });

  it("rejects gateway records that do not belong to the offered local interface", async () => {
    const exchange = vi.fn<GatewayUdpExchange>();
    const mapper = new PcpNatPmpPortMapper({
      gatewayDiscovery: async () => [GATEWAY],
      gatewayValidator: () => false,
      exchange,
      lifetimeSeconds: 60,
      timeoutMs: 100,
      attempts: 1,
      nonceFactory: () => NONCE,
    });
    await expect(mapper.mapTcpPort(input())).resolves.toBeNull();
    expect(exchange).not.toHaveBeenCalled();
  });
});

describe("native port mapper selection", () => {
  it("uses deterministic order, cleans invalid evidence and stops at the first valid mapping", async () => {
    const invalidClose = vi.fn(async () => undefined);
    const first = fakeMapper(null);
    const invalid = fakeMapper({
      externalHost: "100.64.1.2",
      externalPort: 40_000,
      close: invalidClose,
    });
    const validMapping = mapping("8.8.4.4", 40_001);
    const valid = fakeMapper(validMapping);
    const unused = fakeMapper(mapping("1.1.1.1", 40_002));
    const composite = new CompositeTcpPortMapper([first, invalid, valid, unused]);

    await expect(composite.mapTcpPort(input())).resolves.toBe(validMapping);
    expect(first.mapTcpPort).toHaveBeenCalledTimes(1);
    expect(invalid.mapTcpPort).toHaveBeenCalledTimes(1);
    expect(invalidClose).toHaveBeenCalledTimes(1);
    expect(valid.mapTcpPort).toHaveBeenCalledTimes(1);
    expect(unused.mapTcpPort).not.toHaveBeenCalled();
  });

  it("keeps relay eligibility when every native mapper fails", async () => {
    const composite = new CompositeTcpPortMapper([
      fakeMapper(null),
      { mapTcpPort: vi.fn(async () => { throw new Error("gateway_timeout"); }) },
    ]);
    await expect(composite.mapTcpPort(input())).resolves.toBeNull();
  });
});

function mapperWith(exchange: GatewayUdpExchange): PcpNatPmpPortMapper {
  return new PcpNatPmpPortMapper({
    gatewayDiscovery: async () => [GATEWAY],
    gatewayValidator: () => true,
    exchange,
    lifetimeSeconds: 60,
    timeoutMs: 100,
    attempts: 1,
    nonceFactory: () => NONCE,
  });
}

function input() {
  return {
    internalPort: 34_567,
    internalHosts: ["192.168.1.20"],
    description: "Mycellios test",
  };
}

function pcpResponse(
  request: Buffer,
  lifetimeSeconds: number,
  externalHost: string,
  externalPort: number,
  epochSeconds = 100,
): Buffer {
  const response = Buffer.alloc(60);
  response[0] = 2;
  response[1] = 0x81;
  response.writeUInt32BE(lifetimeSeconds, 4);
  response.writeUInt32BE(epochSeconds, 8);
  request.subarray(24, 36).copy(response, 24);
  response[36] = 6;
  response.writeUInt16BE(request.readUInt16BE(40), 40);
  response.writeUInt16BE(externalPort, 42);
  ipv4Mapped(externalHost).copy(response, 44);
  return response;
}

function natPmpAddressResponse(externalHost: string): Buffer {
  const response = Buffer.alloc(12);
  response[1] = 0x80;
  response.writeUInt32BE(100, 4);
  Buffer.from(externalHost.split(".").map(Number)).copy(response, 8);
  return response;
}

function natPmpMapResponse(
  internalPort: number,
  externalPort: number,
  lifetimeSeconds: number,
): Buffer {
  const response = Buffer.alloc(16);
  response[1] = 0x82;
  response.writeUInt32BE(100, 4);
  response.writeUInt16BE(internalPort, 8);
  response.writeUInt16BE(externalPort, 10);
  response.writeUInt32BE(lifetimeSeconds, 12);
  return response;
}

function ipv4Mapped(host: string): Buffer {
  const result = Buffer.alloc(16);
  result[10] = 0xff;
  result[11] = 0xff;
  Buffer.from(host.split(".").map(Number)).copy(result, 12);
  return result;
}

function fakeMapper(result: TcpPortMapping | null): TcpPortMapper {
  return { mapTcpPort: vi.fn(async () => result) };
}

function mapping(externalHost: string, externalPort: number): TcpPortMapping {
  return {
    externalHost,
    externalPort,
    close: vi.fn(async () => undefined),
  };
}
