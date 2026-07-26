import { execFile } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createSocket } from "node:dgram";
import { readFile } from "node:fs/promises";
import { networkInterfaces } from "node:os";
import { isAbsolute, join } from "node:path";
import {
  isPrivateIpv4,
  isPublicIpv4,
  type TcpPortMapper,
  type TcpPortMapping,
} from "./upnp-port-mapper.js";

const NAT_TRAVERSAL_PORT = 5_351;
const PCP_VERSION = 2;
const PCP_MAP_OPCODE = 1;
const NAT_PMP_VERSION = 0;
const NAT_PMP_EXTERNAL_ADDRESS_OPCODE = 0;
const NAT_PMP_TCP_MAP_OPCODE = 2;
const DEFAULT_TIMEOUT_MS = 350;
const DEFAULT_ATTEMPTS = 2;
const DEFAULT_LIFETIME_SECONDS = 7_200;
const MAX_RESPONSE_BYTES = 1_100;
const MAX_GATEWAYS = 8;
const MAX_LIFETIME_SECONDS = 7 * 24 * 60 * 60;

export interface NativeIpv4Gateway {
  gatewayAddress: string;
  localAddress: string;
  interfaceName: string;
  metric: number;
}

export type NativeGatewayDiscovery = () => Promise<NativeIpv4Gateway[]>;

export interface GatewayUdpExchangeInput {
  gatewayAddress: string;
  localAddress: string;
  request: Buffer;
  timeoutMs: number;
  attempts: number;
  maximumResponseBytes: number;
}

export type GatewayUdpExchange = (
  input: GatewayUdpExchangeInput,
) => Promise<Buffer>;

export interface PcpNatPmpPortMapperOptions {
  gatewayDiscovery?: NativeGatewayDiscovery;
  gatewayValidator?: (gateway: NativeIpv4Gateway) => boolean;
  exchange?: GatewayUdpExchange;
  timeoutMs?: number;
  attempts?: number;
  lifetimeSeconds?: number;
  nonceFactory?: () => Buffer;
}

interface MappingLease {
  externalHost: string;
  externalPort: number;
  lifetimeSeconds: number;
  epochSeconds: number;
}

/**
 * Native PCP/NAT-PMP mapper. PCP is attempted first because its per-mapping
 * nonce binds replies and renewals. NAT-PMP is the bounded legacy fallback.
 */
export class PcpNatPmpPortMapper implements TcpPortMapper {
  private readonly discoverGateways: NativeGatewayDiscovery;
  private readonly exchange: GatewayUdpExchange;
  private readonly validateGateway: (gateway: NativeIpv4Gateway) => boolean;
  private readonly timeoutMs: number;
  private readonly attempts: number;
  private readonly lifetimeSeconds: number;
  private readonly nonceFactory: () => Buffer;

  constructor(options: PcpNatPmpPortMapperOptions = {}) {
    this.discoverGateways = options.gatewayDiscovery ?? discoverNativeIpv4Gateways;
    this.validateGateway = options.gatewayValidator ?? validGateway;
    this.exchange = options.exchange ?? exchangeGatewayUdp;
    this.timeoutMs = boundedInteger(
      options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      100,
      2_000,
      "nat_mapping_timeout_is_invalid",
    );
    this.attempts = boundedInteger(
      options.attempts ?? DEFAULT_ATTEMPTS,
      1,
      3,
      "nat_mapping_attempts_are_invalid",
    );
    this.lifetimeSeconds = boundedInteger(
      options.lifetimeSeconds ?? DEFAULT_LIFETIME_SECONDS,
      30,
      MAX_LIFETIME_SECONDS,
      "nat_mapping_lifetime_is_invalid",
    );
    this.nonceFactory = options.nonceFactory ?? (() => randomBytes(12));
  }

  async mapTcpPort(input: {
    internalPort: number;
    internalHosts: string[];
    description: string;
  }): Promise<TcpPortMapping | null> {
    const internalPort = boundedPort(input.internalPort);
    const internalHosts = new Set(input.internalHosts.filter(isPrivateIpv4));
    if (internalHosts.size === 0) return null;
    const discovered = await this.discoverGateways();
    const gateways = discovered
      .filter((gateway) =>
        internalHosts.has(gateway.localAddress)
        && this.validateGateway(gateway)
      )
      .sort((left, right) => left.metric - right.metric)
      .slice(0, MAX_GATEWAYS);
    for (const gateway of gateways) {
      const pcp = await this.tryPcp(gateway, internalPort);
      if (pcp) return pcp;
      const natPmp = await this.tryNatPmp(gateway, internalPort);
      if (natPmp) return natPmp;
    }
    return null;
  }

  private async tryPcp(
    gateway: NativeIpv4Gateway,
    internalPort: number,
  ): Promise<TcpPortMapping | null> {
    const nonce = this.nonceFactory();
    if (nonce.byteLength !== 12) throw new Error("pcp_nonce_is_invalid");
    try {
      const initial = await this.pcpMap({
        gateway,
        internalPort,
        requestedExternalPort: internalPort,
        suggestedExternalHost: null,
        nonce,
        lifetimeSeconds: this.lifetimeSeconds,
      });
      return new RefreshingPortMapping(
        initial,
        async () => this.pcpMap({
          gateway,
          internalPort,
          requestedExternalPort: initial.externalPort,
          suggestedExternalHost: initial.externalHost,
          nonce,
          lifetimeSeconds: this.lifetimeSeconds,
        }),
        async () => {
          await this.pcpMap({
            gateway,
            internalPort,
            requestedExternalPort: initial.externalPort,
            suggestedExternalHost: initial.externalHost,
            nonce,
            lifetimeSeconds: 0,
          });
        },
      );
    } catch {
      return null;
    }
  }

  private async tryNatPmp(
    gateway: NativeIpv4Gateway,
    internalPort: number,
  ): Promise<TcpPortMapping | null> {
    try {
      const external = await this.natPmpExternalAddress(gateway);
      const initial = await this.natPmpMap({
        gateway,
        internalPort,
        requestedExternalPort: internalPort,
        lifetimeSeconds: this.lifetimeSeconds,
      });
      if (initial.epochSeconds < external.epochSeconds) {
        throw new Error("nat_pmp_gateway_restarted_during_mapping");
      }
      const lease: MappingLease = { ...initial, externalHost: external.externalHost };
      return new RefreshingPortMapping(
        lease,
        async () => {
          const currentExternal = await this.natPmpExternalAddress(gateway);
          const renewed = await this.natPmpMap({
            gateway,
            internalPort,
            requestedExternalPort: lease.externalPort,
            lifetimeSeconds: this.lifetimeSeconds,
          });
          if (renewed.epochSeconds < currentExternal.epochSeconds) {
            throw new Error("nat_pmp_gateway_restarted_during_refresh");
          }
          return { ...renewed, externalHost: currentExternal.externalHost };
        },
        async () => {
          await this.natPmpMap({
            gateway,
            internalPort,
            requestedExternalPort: lease.externalPort,
            lifetimeSeconds: 0,
          });
        },
      );
    } catch {
      return null;
    }
  }

  private async pcpMap(input: {
    gateway: NativeIpv4Gateway;
    internalPort: number;
    requestedExternalPort: number;
    suggestedExternalHost: string | null;
    nonce: Buffer;
    lifetimeSeconds: number;
  }): Promise<MappingLease> {
    const request = encodePcpMapRequest({
      localAddress: input.gateway.localAddress,
      internalPort: input.internalPort,
      requestedExternalPort: input.requestedExternalPort,
      suggestedExternalHost: input.suggestedExternalHost,
      nonce: input.nonce,
      lifetimeSeconds: input.lifetimeSeconds,
    });
    const response = await this.exchangePacket(input.gateway, request);
    return decodePcpMapResponse(response, {
      nonce: input.nonce,
      internalPort: input.internalPort,
      deletion: input.lifetimeSeconds === 0,
    });
  }

  private async natPmpExternalAddress(
    gateway: NativeIpv4Gateway,
  ): Promise<{ externalHost: string; epochSeconds: number }> {
    const response = await this.exchangePacket(
      gateway,
      Buffer.from([NAT_PMP_VERSION, NAT_PMP_EXTERNAL_ADDRESS_OPCODE]),
    );
    return decodeNatPmpExternalAddress(response);
  }

  private async natPmpMap(input: {
    gateway: NativeIpv4Gateway;
    internalPort: number;
    requestedExternalPort: number;
    lifetimeSeconds: number;
  }): Promise<Omit<MappingLease, "externalHost">> {
    const request = Buffer.alloc(12);
    request[0] = NAT_PMP_VERSION;
    request[1] = NAT_PMP_TCP_MAP_OPCODE;
    request.writeUInt16BE(boundedPort(input.internalPort), 4);
    request.writeUInt16BE(boundedPort(input.requestedExternalPort), 6);
    request.writeUInt32BE(input.lifetimeSeconds, 8);
    const response = await this.exchangePacket(input.gateway, request);
    return decodeNatPmpMapResponse(response, {
      internalPort: input.internalPort,
      deletion: input.lifetimeSeconds === 0,
    });
  }

  private exchangePacket(
    gateway: NativeIpv4Gateway,
    request: Buffer,
  ): Promise<Buffer> {
    return this.exchange({
      gatewayAddress: gateway.gatewayAddress,
      localAddress: gateway.localAddress,
      request,
      timeoutMs: this.timeoutMs,
      attempts: this.attempts,
      maximumResponseBytes: MAX_RESPONSE_BYTES,
    });
  }
}

class RefreshingPortMapping implements TcpPortMapping {
  readonly externalHost: string;
  readonly externalPort: number;
  private timer: NodeJS.Timeout | null = null;
  private expiresAt: number;
  private epochSeconds: number;
  private closed = false;
  private readonly invalidationListeners = new Set<() => void>();

  constructor(
    initial: MappingLease,
    private readonly renew: () => Promise<MappingLease>,
    private readonly remove: () => Promise<void>,
  ) {
    this.externalHost = initial.externalHost;
    this.externalPort = initial.externalPort;
    this.epochSeconds = initial.epochSeconds;
    this.expiresAt = Date.now() + initial.lifetimeSeconds * 1_000;
    this.scheduleRefresh(initial.lifetimeSeconds);
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    await this.remove();
  }

  onInvalidated(listener: () => void): () => void {
    if (this.closed) {
      queueMicrotask(listener);
      return () => undefined;
    }
    this.invalidationListeners.add(listener);
    return () => this.invalidationListeners.delete(listener);
  }

  private scheduleRefresh(lifetimeSeconds: number): void {
    if (this.closed || lifetimeSeconds <= 0) return;
    const delayMs = Math.max(1_000, Math.floor(lifetimeSeconds * 500));
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.refresh();
    }, delayMs);
    this.timer.unref();
  }

  private async refresh(): Promise<void> {
    if (this.closed) return;
    try {
      const renewed = await this.renew();
      if (
        renewed.externalHost !== this.externalHost
        || renewed.externalPort !== this.externalPort
        || renewed.lifetimeSeconds <= 0
        || renewed.epochSeconds < this.epochSeconds
      ) {
        this.closed = true;
        await this.remove().catch(() => undefined);
        this.notifyInvalidated();
        return;
      }
      this.expiresAt = Date.now() + renewed.lifetimeSeconds * 1_000;
      this.epochSeconds = renewed.epochSeconds;
      this.scheduleRefresh(renewed.lifetimeSeconds);
    } catch {
      const remainingMs = this.expiresAt - Date.now();
      if (remainingMs <= 1_000) {
        this.closed = true;
        this.notifyInvalidated();
        return;
      }
      const retryMs = Math.max(1_000, Math.min(5_000, Math.floor(remainingMs / 2)));
      this.timer = setTimeout(() => {
        this.timer = null;
        void this.refresh();
      }, retryMs);
      this.timer.unref();
    }
  }

  private notifyInvalidated(): void {
    for (const listener of this.invalidationListeners) {
      try {
        listener();
      } catch {
        // Lease state must not depend on an observer.
      }
    }
    this.invalidationListeners.clear();
  }
}

export async function discoverNativeIpv4Gateways(): Promise<NativeIpv4Gateway[]> {
  const interfaces = networkInterfaces();
  let candidates: NativeIpv4Gateway[] = [];
  try {
    if (process.platform === "linux") {
      const routeTable = await readFile("/proc/net/route", "utf8");
      if (Buffer.byteLength(routeTable, "utf8") > 64 * 1024) return [];
      candidates = parseLinuxRoutes(routeTable, interfaces);
    } else if (process.platform === "win32") {
      const systemRoot = process.env.SystemRoot;
      if (!systemRoot || !isAbsolute(systemRoot)) return [];
      candidates = parseWindowsRoutes(
        await execFileText(join(systemRoot, "System32", "route.exe"), ["PRINT", "-4"]),
        interfaces,
      );
    } else if (process.platform === "darwin") {
      candidates = parseDarwinRoute(
        await execFileText("/sbin/route", ["-n", "get", "default"]),
        interfaces,
      );
    }
  } catch {
    return [];
  }
  return candidates
    .filter(validGateway)
    .sort((left, right) => left.metric - right.metric)
    .slice(0, MAX_GATEWAYS);
}

export async function exchangeGatewayUdp(
  input: GatewayUdpExchangeInput,
): Promise<Buffer> {
  if (
    !isPrivateIpv4(input.gatewayAddress)
    || !isPrivateIpv4(input.localAddress)
    || !sameInterfaceSubnet(input.gatewayAddress, input.localAddress)
    || input.request.byteLength < 2
    || input.request.byteLength > MAX_RESPONSE_BYTES
  ) throw new Error("nat_gateway_exchange_is_invalid");
  const timeoutMs = boundedInteger(
    input.timeoutMs,
    100,
    2_000,
    "nat_gateway_timeout_is_invalid",
  );
  const attempts = boundedInteger(
    input.attempts,
    1,
    3,
    "nat_gateway_attempts_are_invalid",
  );
  const maximumResponseBytes = boundedInteger(
    input.maximumResponseBytes,
    16,
    MAX_RESPONSE_BYTES,
    "nat_gateway_response_limit_is_invalid",
  );
  return new Promise<Buffer>((resolve, reject) => {
    const socket = createSocket("udp4");
    let timer: NodeJS.Timeout | null = null;
    let sent = 0;
    let settled = false;
    const finish = (error: Error | null, response?: Buffer) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      try {
        socket.close();
      } catch {
        // An early bind error can leave a datagram socket not yet running.
      }
      if (error) reject(error);
      else resolve(response!);
    };
    const send = () => {
      sent += 1;
      socket.send(input.request, (error) => {
        if (error) finish(error);
      });
      timer = setTimeout(() => {
        timer = null;
        if (sent >= attempts) {
          finish(new Error("nat_gateway_request_timed_out"));
        } else {
          send();
        }
      }, timeoutMs);
      timer.unref();
    };
    socket.once("error", (error) => finish(error));
    socket.on("message", (response, remote) => {
      if (
        remote.address !== input.gatewayAddress
        || remote.port !== NAT_TRAVERSAL_PORT
        || response.byteLength > maximumResponseBytes
      ) {
        finish(new Error("nat_gateway_response_is_invalid"));
        return;
      }
      finish(null, Buffer.from(response));
    });
    socket.bind(0, input.localAddress, () => {
      socket.connect(NAT_TRAVERSAL_PORT, input.gatewayAddress, send);
    });
  });
}

export function encodePcpMapRequest(input: {
  localAddress: string;
  internalPort: number;
  requestedExternalPort: number;
  suggestedExternalHost: string | null;
  nonce: Buffer;
  lifetimeSeconds: number;
}): Buffer {
  if (input.nonce.byteLength !== 12 || !isPrivateIpv4(input.localAddress)) {
    throw new Error("pcp_request_identity_is_invalid");
  }
  const request = Buffer.alloc(60);
  request[0] = PCP_VERSION;
  request[1] = PCP_MAP_OPCODE;
  request.writeUInt32BE(
    boundedInteger(input.lifetimeSeconds, 0, MAX_LIFETIME_SECONDS, "pcp_lifetime_is_invalid"),
    4,
  );
  encodeIpv4Mapped(input.localAddress).copy(request, 8);
  input.nonce.copy(request, 24);
  request[36] = 6;
  request.writeUInt16BE(boundedPort(input.internalPort), 40);
  request.writeUInt16BE(boundedPort(input.requestedExternalPort), 42);
  if (input.suggestedExternalHost) {
    if (!isPublicIpv4(input.suggestedExternalHost)) {
      throw new Error("pcp_suggested_address_is_invalid");
    }
    encodeIpv4Mapped(input.suggestedExternalHost).copy(request, 44);
  }
  return request;
}

export function decodePcpMapResponse(
  response: Buffer,
  expected: {
    nonce: Buffer;
    internalPort: number;
    deletion: boolean;
  },
): MappingLease {
  if (
    response.byteLength !== 60
    || response[0] !== PCP_VERSION
    || response[1] !== (0x80 | PCP_MAP_OPCODE)
    || response[2] !== 0
    || response[3] !== 0
    || response.subarray(12, 24).some((byte) => byte !== 0)
    || expected.nonce.byteLength !== 12
    || !response.subarray(24, 36).equals(expected.nonce)
    || response[36] !== 6
    || response.subarray(37, 40).some((byte) => byte !== 0)
    || response.readUInt16BE(40) !== boundedPort(expected.internalPort)
  ) throw new Error("pcp_map_response_is_invalid");
  const lifetimeSeconds = response.readUInt32BE(4);
  const epochSeconds = response.readUInt32BE(8);
  if (
    (expected.deletion && lifetimeSeconds !== 0)
    || (!expected.deletion && (lifetimeSeconds < 1 || lifetimeSeconds > MAX_LIFETIME_SECONDS))
  ) throw new Error("pcp_map_lifetime_is_invalid");
  const externalPort = response.readUInt16BE(42);
  const externalHost = decodeIpv4Mapped(response.subarray(44, 60));
  if (
    externalPort < 1
    || !isPublicIpv4(externalHost)
  ) throw new Error("pcp_map_endpoint_is_invalid");
  return { externalHost, externalPort, lifetimeSeconds, epochSeconds };
}

export function decodeNatPmpExternalAddress(
  response: Buffer,
): { externalHost: string; epochSeconds: number } {
  if (
    response.byteLength !== 12
    || response[0] !== NAT_PMP_VERSION
    || response[1] !== (0x80 | NAT_PMP_EXTERNAL_ADDRESS_OPCODE)
    || response.readUInt16BE(2) !== 0
  ) throw new Error("nat_pmp_external_address_response_is_invalid");
  const address = [...response.subarray(8, 12)].join(".");
  if (!isPublicIpv4(address)) throw new Error("nat_pmp_external_address_is_not_public");
  return {
    externalHost: address,
    epochSeconds: response.readUInt32BE(4),
  };
}

export function decodeNatPmpMapResponse(
  response: Buffer,
  expected: { internalPort: number; deletion: boolean },
): Omit<MappingLease, "externalHost"> {
  if (
    response.byteLength !== 16
    || response[0] !== NAT_PMP_VERSION
    || response[1] !== (0x80 | NAT_PMP_TCP_MAP_OPCODE)
    || response.readUInt16BE(2) !== 0
    || response.readUInt16BE(8) !== boundedPort(expected.internalPort)
  ) throw new Error("nat_pmp_map_response_is_invalid");
  const externalPort = response.readUInt16BE(10);
  const lifetimeSeconds = response.readUInt32BE(12);
  const epochSeconds = response.readUInt32BE(4);
  if (
    (!expected.deletion && externalPort < 1)
    || (expected.deletion && lifetimeSeconds !== 0)
    || (!expected.deletion && (lifetimeSeconds < 1 || lifetimeSeconds > MAX_LIFETIME_SECONDS))
  ) throw new Error("nat_pmp_map_endpoint_is_invalid");
  return { externalPort, lifetimeSeconds, epochSeconds };
}

function validGateway(gateway: NativeIpv4Gateway): boolean {
  return (
    isPrivateIpv4(gateway.gatewayAddress)
    && isPrivateIpv4(gateway.localAddress)
    && gateway.gatewayAddress !== gateway.localAddress
    && typeof gateway.interfaceName === "string"
    && gateway.interfaceName.length > 0
    && Number.isSafeInteger(gateway.metric)
    && gateway.metric >= 0
    && sameInterfaceSubnet(gateway.gatewayAddress, gateway.localAddress, gateway.interfaceName)
  );
}

function sameInterfaceSubnet(
  gatewayAddress: string,
  localAddress: string,
  interfaceName?: string,
): boolean {
  const entries = interfaceName
    ? (networkInterfaces()[interfaceName] ?? [])
    : Object.values(networkInterfaces()).flatMap((addresses) => addresses ?? []);
  const local = entries.find((entry) =>
    entry.family === "IPv4" && entry.address === localAddress
  );
  if (!local) return false;
  const gateway = ipv4Number(gatewayAddress);
  const address = ipv4Number(local.address);
  const mask = ipv4Number(local.netmask);
  return gateway !== null
    && address !== null
    && mask !== null
    && (gateway & mask) === (address & mask);
}

function parseLinuxRoutes(
  text: string,
  interfaces: NodeJS.Dict<import("node:os").NetworkInterfaceInfo[]>,
): NativeIpv4Gateway[] {
  return text.split(/\r?\n/).slice(1).flatMap((line) => {
    const fields = line.trim().split(/\s+/);
    if (fields.length < 8 || fields[1] !== "00000000") return [];
    const interfaceName = fields[0]!;
    const gatewayAddress = littleEndianHexIpv4(fields[2]!);
    const flags = Number.parseInt(fields[3]!, 16);
    const metric = Number.parseInt(fields[6]!, 10);
    const localAddress = gatewayAddress
      ? privateAddressForInterface(interfaces, interfaceName, gatewayAddress)
      : null;
    return (
      gatewayAddress
      && localAddress
      && (flags & 0x3) === 0x3
      && Number.isSafeInteger(metric)
    )
      ? [{ gatewayAddress, localAddress, interfaceName, metric }]
      : [];
  });
}

function parseWindowsRoutes(
  text: string,
  interfaces: NodeJS.Dict<import("node:os").NetworkInterfaceInfo[]>,
): NativeIpv4Gateway[] {
  const byAddress = interfaceAddressIndex(interfaces);
  return text.split(/\r?\n/).flatMap((line) => {
    const match =
      /^\s*0\.0\.0\.0\s+0\.0\.0\.0\s+(\d{1,3}(?:\.\d{1,3}){3})\s+(\d{1,3}(?:\.\d{1,3}){3})\s+(\d+)\s*$/
        .exec(line);
    if (!match) return [];
    const gatewayAddress = match[1]!;
    const localAddress = match[2]!;
    const metric = Number(match[3]);
    const interfaceName = byAddress.get(localAddress);
    return interfaceName && Number.isSafeInteger(metric)
      ? [{ gatewayAddress, localAddress, interfaceName, metric }]
      : [];
  });
}

function parseDarwinRoute(
  text: string,
  interfaces: NodeJS.Dict<import("node:os").NetworkInterfaceInfo[]>,
): NativeIpv4Gateway[] {
  const gatewayAddress = /^\s*gateway:\s*(\d{1,3}(?:\.\d{1,3}){3})\s*$/mi.exec(text)?.[1];
  const interfaceName = /^\s*interface:\s*(\S+)\s*$/mi.exec(text)?.[1];
  if (!gatewayAddress || !interfaceName) return [];
  const localAddress = privateAddressForInterface(interfaces, interfaceName, gatewayAddress);
  return localAddress
    ? [{ gatewayAddress, localAddress, interfaceName, metric: 0 }]
    : [];
}

function privateAddressForInterface(
  interfaces: NodeJS.Dict<import("node:os").NetworkInterfaceInfo[]>,
  interfaceName: string,
  gatewayAddress: string,
): string | null {
  return (interfaces[interfaceName] ?? []).find((entry) =>
    entry.family === "IPv4"
    && !entry.internal
    && isPrivateIpv4(entry.address)
    && sameSubnet(entry.address, gatewayAddress, entry.netmask)
  )?.address ?? null;
}

function interfaceAddressIndex(
  interfaces: NodeJS.Dict<import("node:os").NetworkInterfaceInfo[]>,
): Map<string, string> {
  const result = new Map<string, string>();
  for (const [name, entries] of Object.entries(interfaces)) {
    for (const entry of entries ?? []) {
      if (entry.family === "IPv4" && !entry.internal) result.set(entry.address, name);
    }
  }
  return result;
}

function littleEndianHexIpv4(value: string): string | null {
  if (!/^[0-9A-Fa-f]{8}$/.test(value)) return null;
  return [6, 4, 2, 0].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16)).join(".");
}

function sameSubnet(left: string, right: string, netmask: string): boolean {
  const leftNumber = ipv4Number(left);
  const rightNumber = ipv4Number(right);
  const maskNumber = ipv4Number(netmask);
  return leftNumber !== null
    && rightNumber !== null
    && maskNumber !== null
    && (leftNumber & maskNumber) === (rightNumber & maskNumber);
}

function ipv4Number(value: string): number | null {
  const parts = value.split(".");
  if (
    parts.length !== 4
    || parts.some((part) => !/^(?:0|[1-9]\d{0,2})$/.test(part))
  ) return null;
  const octets = parts.map(Number);
  if (octets.some((part) => part < 0 || part > 255)) return null;
  return (
    ((octets[0]! << 24) >>> 0)
    + (octets[1]! << 16)
    + (octets[2]! << 8)
    + octets[3]!
  ) >>> 0;
}

function encodeIpv4Mapped(address: string): Buffer {
  const value = ipv4Number(address);
  if (value === null) throw new Error("pcp_ipv4_address_is_invalid");
  const result = Buffer.alloc(16);
  result[10] = 0xff;
  result[11] = 0xff;
  result.writeUInt32BE(value, 12);
  return result;
}

function decodeIpv4Mapped(value: Buffer): string {
  if (
    value.byteLength !== 16
    || value.subarray(0, 10).some((byte) => byte !== 0)
    || value[10] !== 0xff
    || value[11] !== 0xff
  ) throw new Error("pcp_external_address_is_not_ipv4");
  return [...value.subarray(12, 16)].join(".");
}

function execFileText(executable: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(
      executable,
      args,
      {
        encoding: "utf8",
        timeout: 1_500,
        maxBuffer: 64 * 1024,
        windowsHide: true,
      },
      (error, stdout) => {
        if (error) reject(error);
        else resolve(stdout);
      },
    );
  });
}

function boundedPort(value: number): number {
  return boundedInteger(value, 1, 65_535, "nat_mapping_port_is_invalid");
}

function boundedInteger(value: number, minimum: number, maximum: number, code: string): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(code);
  return value;
}
