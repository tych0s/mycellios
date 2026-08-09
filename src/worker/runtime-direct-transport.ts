import { once } from "node:events";
import { networkInterfaces } from "node:os";
import { createConnection, type Socket } from "node:net";
import {
  DIRECT_TRANSPORT_PROTOCOL,
  DirectSecureChannel,
  DirectSecureServer,
  type DirectSessionGrant,
} from "../transport/direct-secure-channel.js";
import {
  DirectRuntimeMux,
  type DirectRuntimeStream,
} from "../transport/direct-runtime-mux.js";
import {
  isPrivateIpv4,
  isPublicIpv4,
  type TcpPortMapper,
  type TcpPortMapping,
} from "./upnp-port-mapper.js";
import { NativeTcpPortMapper } from "./native-port-mapper.js";

const READY_MARKER = Buffer.from("MYCELLIOS-DIRECT-READY/1", "utf8");
const DEFAULT_CONNECT_TIMEOUT_MS = 2_000;
const DEFAULT_MAX_SESSION_BYTES = 512 * 1024 * 1024;
const DEFAULT_MAX_SESSIONS = 64;
const DEFAULT_MAX_SESSION_MS = 4 * 60 * 60 * 1_000;
const MAX_CANDIDATES = 8;
const MAX_SOCKET_BUFFERED_BYTES = 8 * 1024 * 1024;

export interface DirectTransportCandidate {
  host: string;
  port: number;
  scope: "lan" | "configured" | "public-mapped";
}

export interface DirectTransportAdvertisement {
  protocol: typeof DIRECT_TRANSPORT_PROTOCOL;
  /**
   * The destination acknowledges its local commit before the coordinator
   * releases the source socket. Absent on legacy workers, which must use relay.
   */
  commitAck: "destination-v1";
  candidates: DirectTransportCandidate[];
  maxSessions: number;
  maxSessionBytes: number;
}

export interface RuntimeDirectTransportOptions {
  enabled?: boolean;
  listenHost?: string;
  listenPort?: number;
  candidateHosts?: string[];
  /** Opt in to a native UPnP IGD mapping. Failure leaves relay available. */
  publicPortMapping?: boolean;
  /** Test/platform injection point. Supplying a mapper also opts in. */
  portMapper?: TcpPortMapper;
  connectTimeoutMs?: number;
  maxSessions?: number;
  maxSessionBytes?: number;
  maxSessionMs?: number;
}

export interface RuntimeDirectTransportSnapshot {
  streamId: string;
  sourceNodeId: string;
  destinationNodeId: string;
  targetPort: number;
  mode: "direct";
  state: "negotiating" | "active" | "closed";
  bytesTx: number;
  bytesRx: number;
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
  connectRttMs: number | null;
}

interface DestinationOffer {
  streamId: string;
  grant: DirectSessionGrant;
  expires: NodeJS.Timeout;
}

interface DirectRecord {
  streamId: string;
  role: "source" | "destination";
  grant: DirectSessionGrant;
  socket: Socket;
  channel: DirectSecureChannel;
  mux: DirectRuntimeMux;
  stream: DirectRuntimeStream;
  committed: boolean;
  bytesTx: number;
  bytesRx: number;
  createdAt: number;
  connectedAt: number | null;
  endedAt: number | null;
  connectRttMs: number | null;
  lastTelemetryAt: number;
  lastTelemetryBytes: number;
  writeQueue: Promise<void>;
  lifetime: NodeJS.Timeout;
  closed: boolean;
}

export interface RuntimeDirectTransportCallbacks {
  send(type: string, payload: unknown): boolean | void;
  isTargetPortAuthorized(targetPort: number): boolean;
  onSourceCommitted(streamId: string): void;
  onSourceFailed(streamId: string, error: Error): void;
  onSourceClosed(streamId: string, error?: Error): void;
  onAdvertisementChanged?(advertisement: DirectTransportAdvertisement): void;
}

/**
 * Native peer transport for runtime streams.
 *
 * The coordinator is only a signaling authority. It issues the bound one-time
 * grant and commits the route. No application byte is accepted before that
 * commit, so a failed candidate can still fall back to the authenticated relay
 * without ambiguous delivery.
 */
export class RuntimeDirectTransport {
  private readonly offers = new Map<string, DestinationOffer>();
  private readonly records = new Map<string, DirectRecord>();
  private readonly completed: RuntimeDirectTransportSnapshot[] = [];
  private readonly connectTimeoutMs: number;
  private readonly maxSessions: number;
  private readonly maxSessionBytes: number;
  private readonly maxSessionMs: number;
  private server: DirectSecureServer | null = null;
  private advertisement: DirectTransportAdvertisement | null = null;
  private publicMapping: TcpPortMapping | null = null;
  private unsubscribePublicMappingInvalidation: (() => void) | null = null;

  constructor(
    private readonly nodeId: string,
    private readonly callbacks: RuntimeDirectTransportCallbacks,
    private readonly options: RuntimeDirectTransportOptions = {},
  ) {
    this.connectTimeoutMs = boundedInteger(
      options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS,
      250,
      15_000,
      "direct_connect_timeout_is_invalid",
    );
    this.maxSessions = boundedInteger(
      options.maxSessions ?? DEFAULT_MAX_SESSIONS,
      1,
      256,
      "direct_session_limit_is_invalid",
    );
    this.maxSessionBytes = boundedInteger(
      options.maxSessionBytes ?? DEFAULT_MAX_SESSION_BYTES,
      1024 * 1024,
      16 * 1024 * 1024 * 1024,
      "direct_session_byte_limit_is_invalid",
    );
    this.maxSessionMs = boundedInteger(
      options.maxSessionMs ?? DEFAULT_MAX_SESSION_MS,
      10_000,
      24 * 60 * 60 * 1_000,
      "direct_session_lifetime_is_invalid",
    );
  }

  async start(): Promise<DirectTransportAdvertisement | null> {
    if (this.options.enabled === false) return null;
    if (this.server) {
      return this.advertisement ? structuredClone(this.advertisement) : null;
    }
    const server = new DirectSecureServer({
      host: this.options.listenHost ?? "0.0.0.0",
      port: this.options.listenPort ?? 0,
      maximumFrameBytes: 128 * 1024,
      handshakeTimeoutMs: this.connectTimeoutMs,
      resolveGrant: (connectionId) => this.offers.get(connectionId)?.grant ?? null,
      onChannel: (channel, grant) => {
        void this.acceptDestinationChannel(channel, grant);
      },
    });
    let address: { host: string; port: number };
    try {
      address = await server.listen();
    } catch {
      await server.close().catch(() => undefined);
      return null;
    }
    let candidates = directCandidates(
      address.port,
      this.options.candidateHosts,
      this.options.listenHost,
    );
    if (this.options.publicPortMapping === true || this.options.portMapper) {
      const internalHosts = Array.from(new Set(
        candidates
          .map((candidate) => candidate.host)
          .filter(isPrivateIpv4),
      ));
      if (internalHosts.length > 0) {
        try {
          const mapping = await (this.options.portMapper ?? new NativeTcpPortMapper())
            .mapTcpPort({
              internalPort: address.port,
              internalHosts,
              description: `Mycellios ${this.nodeId}`,
            });
          if (mapping) {
            if (
              !isPublicIpv4(mapping.externalHost)
              || !Number.isSafeInteger(mapping.externalPort)
              || mapping.externalPort < 1
              || mapping.externalPort > 65_535
            ) {
              await mapping.close().catch(() => undefined);
            } else {
              this.publicMapping = mapping;
              this.unsubscribePublicMappingInvalidation = mapping.onInvalidated?.(() => {
                this.invalidatePublicMapping(mapping);
              }) ?? null;
              candidates = candidates.slice(0, MAX_CANDIDATES - 1);
              addCandidate(
                candidates,
                mapping.externalHost,
                mapping.externalPort,
                "public-mapped",
              );
            }
          }
        } catch {
          // A mapping is an optimization, never a reachability claim. The
          // authenticated coordinator relay remains the fallback.
        }
      }
    }
    if (candidates.length === 0) {
      await this.closePublicMapping();
      await server.close();
      return null;
    }
    this.server = server;
    this.advertisement = {
      protocol: DIRECT_TRANSPORT_PROTOCOL,
      commitAck: "destination-v1",
      candidates,
      maxSessions: this.maxSessions,
      maxSessionBytes: this.maxSessionBytes,
    };
    return structuredClone(this.advertisement);
  }

  installDestinationOffer(streamId: string, grant: DirectSessionGrant): void {
    this.validateGrantForRole(streamId, grant, "destination");
    if (!this.server || !this.advertisement) {
      this.rejectOffer(streamId, grant.connectionId, "direct_listener_unavailable");
      return;
    }
    if (!this.callbacks.isTargetPortAuthorized(grant.targetPort)) {
      this.rejectOffer(streamId, grant.connectionId, "direct_target_not_authorized");
      return;
    }
    if (this.offers.size + this.records.size >= this.maxSessions) {
      this.rejectOffer(streamId, grant.connectionId, "direct_session_capacity_exceeded");
      return;
    }
    if (this.offers.has(grant.connectionId) || this.records.has(streamId)) {
      this.rejectOffer(streamId, grant.connectionId, "direct_offer_is_duplicate");
      return;
    }
    const expires = setTimeout(() => {
      if (!this.offers.delete(grant.connectionId)) return;
      this.rejectOffer(streamId, grant.connectionId, "direct_offer_expired");
    }, Math.max(1, grant.expiresAt - Date.now()));
    expires.unref();
    this.offers.set(grant.connectionId, { streamId, grant, expires });
    this.callbacks.send("runtime.direct.ready", {
      streamId,
      connectionId: grant.connectionId,
    });
  }

  async connectSource(input: {
    streamId: string;
    destinationNodeId: string;
    grant: DirectSessionGrant;
    candidates: DirectTransportCandidate[];
    socket: Socket;
    timeoutMs: number;
  }): Promise<void> {
    this.validateGrantForRole(input.streamId, input.grant, "source");
    if (
      input.grant.destinationNodeId !== input.destinationNodeId
      || input.grant.targetPort < 1
      || input.candidates.length < 1
      || input.candidates.length > MAX_CANDIDATES
      || this.records.has(input.streamId)
      || this.records.size >= this.maxSessions
    ) {
      this.rejectSource(input.streamId, input.grant.connectionId, "direct_connect_request_is_invalid");
      return;
    }
    const timeoutMs = boundedInteger(
      Math.min(input.timeoutMs, this.connectTimeoutMs),
      250,
      15_000,
      "direct_connect_timeout_is_invalid",
    );
    const startedAt = process.hrtime.bigint();
    let lastError = new Error("direct_candidate_unreachable");
    for (const candidate of orderDirectCandidates(input.candidates)) {
      let channel: DirectSecureChannel | null = null;
      let provisional: DirectRecord | null = null;
      try {
        channel = await DirectSecureChannel.connect({
          host: validateCandidateHost(candidate),
          port: boundedPort(candidate.port),
          grant: input.grant,
          timeoutMs,
          maximumFrameBytes: 128 * 1024,
        });
        const mux = new DirectRuntimeMux(channel, {
          maximumStreams: 1,
          receiveWindowBytes: 4 * 1024 * 1024,
        });
        const stream = await mux.openStream(input.streamId);
        const ready = deferred<void>();
        let markerReceived = false;
        const record = this.createRecord({
          streamId: input.streamId,
          role: "source",
          grant: input.grant,
          socket: input.socket,
          channel,
          mux,
          stream,
          connectRttMs: null,
        });
        provisional = record;
        this.records.set(record.streamId, record);
        stream.setDataHandler(async (data) => {
          if (!markerReceived) {
            if (!data.equals(READY_MARKER)) {
              throw new Error("direct_ready_marker_is_invalid");
            }
            markerReceived = true;
            ready.resolve();
            return;
          }
          await this.writeToSocket(record, data);
          record.bytesRx = checkedBytes(record.bytesRx, data.byteLength, this.maxSessionBytes);
          this.reportTelemetry(record);
        });
        stream.once("end", () => this.finishRecord(record));
        stream.once("reset", (error) => {
          if (!record.committed) ready.reject(error);
          else this.finishRecord(record, error);
        });
        mux.on("error", (error) => {
          if (!record.committed) ready.reject(error);
          else this.finishRecord(record, error);
        });
        await withTimeout(ready.promise, timeoutMs, "direct_target_ready_timeout");
        record.connectRttMs = Number(process.hrtime.bigint() - startedAt) / 1_000_000;
        this.callbacks.send("runtime.direct.established", {
          streamId: record.streamId,
          connectionId: record.grant.connectionId,
          connectRttMs: Math.max(Number.EPSILON, record.connectRttMs),
        });
        return;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (provisional) {
          this.abortProvisional(provisional, true);
          this.rejectSource(input.streamId, input.grant.connectionId, lastError.message);
          return;
        }
        channel?.destroy();
      }
    }
    this.rejectSource(input.streamId, input.grant.connectionId, lastError.message);
  }

  commit(streamId: string, connectionId: string): DirectRecord["role"] | false {
    const record = this.records.get(streamId);
    if (!record || record.grant.connectionId !== connectionId || record.committed) return false;
    record.committed = true;
    record.connectedAt = Date.now();
    record.socket.resume();
    if (record.role === "source") this.callbacks.onSourceCommitted(streamId);
    return record.role;
  }

  writeSource(streamId: string, data: Buffer): boolean {
    const record = this.records.get(streamId);
    if (!record || record.role !== "source" || !record.committed || record.closed) return false;
    this.queueWrite(record, data);
    return true;
  }

  cancel(streamId: string, connectionId: string): void {
    const offer = this.offers.get(connectionId);
    if (offer?.streamId === streamId) {
      clearTimeout(offer.expires);
      this.offers.delete(connectionId);
    }
    const record = this.records.get(streamId);
    if (record?.grant.connectionId === connectionId) {
      if (record.role === "source" && !record.committed) {
        this.abortProvisional(record, true);
      } else {
        this.finishRecord(record, new Error("direct_route_cancelled"));
      }
    }
  }

  closeSource(streamId: string, error?: Error): void {
    const record = this.records.get(streamId);
    if (record?.role === "source") this.finishRecord(record, error);
  }

  snapshot(): RuntimeDirectTransportSnapshot[] {
    const active = [...this.records.values()].map((record) => this.recordSnapshot(record));
    return [...active, ...this.completed].map((snapshot) => ({ ...snapshot }));
  }

  async close(): Promise<void> {
    for (const offer of this.offers.values()) clearTimeout(offer.expires);
    this.offers.clear();
    for (const record of [...this.records.values()]) {
      this.finishRecord(record, new Error("direct_transport_closed"));
    }
    const server = this.server;
    this.server = null;
    this.advertisement = null;
    await this.closePublicMapping();
    await server?.close();
  }

  private async closePublicMapping(): Promise<void> {
    const mapping = this.publicMapping;
    this.publicMapping = null;
    this.unsubscribePublicMappingInvalidation?.();
    this.unsubscribePublicMappingInvalidation = null;
    await mapping?.close().catch(() => undefined);
  }

  private invalidatePublicMapping(mapping: TcpPortMapping): void {
    if (this.publicMapping !== mapping) return;
    this.unsubscribePublicMappingInvalidation?.();
    this.unsubscribePublicMappingInvalidation = null;
    this.publicMapping = null;
    if (!this.advertisement) return;
    this.advertisement = {
      ...this.advertisement,
      candidates: this.advertisement.candidates.filter((candidate) =>
        !(
          candidate.scope === "public-mapped"
          && candidate.host === mapping.externalHost
          && candidate.port === mapping.externalPort
        )
      ),
    };
    this.callbacks.onAdvertisementChanged?.(structuredClone(this.advertisement));
  }

  private async acceptDestinationChannel(
    channel: DirectSecureChannel,
    grant: DirectSessionGrant,
  ): Promise<void> {
    const offer = this.offers.get(grant.connectionId);
    if (
      !offer
      || offer.grant.secret !== grant.secret
      || offer.grant.targetPort !== grant.targetPort
      || this.records.has(offer.streamId)
    ) {
      channel.destroy();
      return;
    }
    clearTimeout(offer.expires);
    this.offers.delete(grant.connectionId);
    const socket = createConnection({ host: "127.0.0.1", port: grant.targetPort });
    socket.setNoDelay(true);
    socket.pause();
    const mux = new DirectRuntimeMux(channel, {
      maximumStreams: 1,
      receiveWindowBytes: 4 * 1024 * 1024,
    });
    const incoming = deferred<DirectRuntimeStream>();
    mux.once("stream", (stream) => incoming.resolve(stream));
    mux.once("error", (error) => incoming.reject(error));
    try {
      const [, stream] = await Promise.all([
        withTimeout(new Promise<void>((resolve, reject) => {
          socket.once("connect", resolve);
          socket.once("error", reject);
        }), this.connectTimeoutMs, "direct_target_connect_timeout"),
        withTimeout(
          incoming.promise,
          this.connectTimeoutMs,
          "direct_stream_open_timeout",
        ),
      ]);
      if (stream.id !== offer.streamId) throw new Error("direct_stream_identity_mismatch");
      const record = this.createRecord({
        streamId: offer.streamId,
        role: "destination",
        grant,
        socket,
        channel,
        mux,
        stream,
        connectRttMs: null,
      });
      stream.setDataHandler(async (data) => {
        await this.writeToSocket(record, data);
        record.bytesRx = checkedBytes(record.bytesRx, data.byteLength, this.maxSessionBytes);
        this.reportTelemetry(record);
      });
      stream.once("end", () => this.finishRecord(record));
      stream.once("reset", (error) => this.finishRecord(record, error));
      mux.on("error", (error) => this.finishRecord(record, error));
      this.records.set(record.streamId, record);
      await stream.write(READY_MARKER);
    } catch (error) {
      socket.destroy();
      channel.destroy();
      this.callbacks.send("runtime.direct.fallback", {
        streamId: offer.streamId,
        connectionId: grant.connectionId,
        reason: sanitizeReason(error),
      });
    }
  }

  private createRecord(input: {
    streamId: string;
    role: "source" | "destination";
    grant: DirectSessionGrant;
    socket: Socket;
    channel: DirectSecureChannel;
    mux: DirectRuntimeMux;
    stream: DirectRuntimeStream;
    connectRttMs: number | null;
  }): DirectRecord {
    const record: DirectRecord = {
      ...input,
      committed: false,
      bytesTx: 0,
      bytesRx: 0,
      createdAt: Date.now(),
      connectedAt: null,
      endedAt: null,
      lastTelemetryAt: 0,
      lastTelemetryBytes: 0,
      writeQueue: Promise.resolve(),
      lifetime: setTimeout(() => undefined, this.maxSessionMs),
      closed: false,
    };
    clearTimeout(record.lifetime);
    record.lifetime = setTimeout(() => {
      this.finishRecord(record, new Error("direct_session_lifetime_exceeded"));
    }, this.maxSessionMs);
    record.lifetime.unref();
    if (input.role === "destination") {
      input.socket.on("data", (data: Buffer) => {
        if (!record.committed) {
          this.finishRecord(record, new Error("direct_data_before_commit"));
          return;
        }
        this.queueWrite(record, data);
      });
    }
    input.socket.once("end", () => this.finishRecord(record));
    input.socket.once("close", () => this.finishRecord(record));
    input.socket.once("error", (error) => this.finishRecord(record, error));
    return record;
  }

  private queueWrite(record: DirectRecord, data: Buffer): void {
    if (record.closed) return;
    record.socket.pause();
    record.writeQueue = record.writeQueue.then(async () => {
      record.bytesTx = checkedBytes(record.bytesTx, data.byteLength, this.maxSessionBytes);
      await record.stream.write(data);
      this.reportTelemetry(record);
      if (record.committed && !record.closed) record.socket.resume();
    }).catch((error: unknown) => {
      this.finishRecord(record, error instanceof Error ? error : new Error(String(error)));
    });
  }

  private async writeToSocket(record: DirectRecord, data: Buffer): Promise<void> {
    if (!record.committed || record.closed) {
      throw new Error("direct_data_before_commit");
    }
    if (record.socket.destroyed) throw new Error("direct_target_socket_closed");
    if (record.socket.writableLength + data.byteLength > MAX_SOCKET_BUFFERED_BYTES) {
      throw new Error("direct_target_socket_backpressure_exceeded");
    }
    if (!record.socket.write(data)) await once(record.socket, "drain");
  }

  private reportTelemetry(record: DirectRecord): void {
    if (!record.committed || record.closed) return;
    const now = Date.now();
    const total = record.bytesTx + record.bytesRx;
    if (
      now - record.lastTelemetryAt < 1_000
      && total - record.lastTelemetryBytes < 1024 * 1024
    ) return;
    record.lastTelemetryAt = now;
    record.lastTelemetryBytes = total;
    this.callbacks.send("runtime.direct.telemetry", {
      streamId: record.streamId,
      connectionId: record.grant.connectionId,
      bytesTx: record.bytesTx,
      bytesRx: record.bytesRx,
    });
  }

  private finishRecord(record: DirectRecord, error?: Error): void {
    if (record.closed) return;
    record.closed = true;
    clearTimeout(record.lifetime);
    this.records.delete(record.streamId);
    record.endedAt = Date.now();
    record.socket.destroy(error);
    void record.mux.close().catch(() => undefined);
    record.channel.destroy();
    this.completed.unshift(this.recordSnapshot(record));
    if (this.completed.length > 256) this.completed.length = 256;
    if (record.committed) {
      this.callbacks.send("runtime.direct.closed", {
        streamId: record.streamId,
        connectionId: record.grant.connectionId,
        bytesTx: record.bytesTx,
        bytesRx: record.bytesRx,
        ...(error ? { reason: sanitizeReason(error) } : {}),
      });
      if (record.role === "source") this.callbacks.onSourceClosed(record.streamId, error);
    } else if (record.role === "source") {
      this.rejectSource(record.streamId, record.grant.connectionId, error?.message ?? "direct_closed");
    }
  }

  private abortProvisional(record: DirectRecord, preserveSourceSocket: boolean): void {
    if (record.closed) return;
    record.closed = true;
    clearTimeout(record.lifetime);
    this.records.delete(record.streamId);
    if (!(preserveSourceSocket && record.role === "source")) {
      record.socket.destroy();
    } else {
      record.socket.pause();
    }
    void record.mux.close().catch(() => undefined);
    record.channel.destroy();
  }

  private recordSnapshot(record: DirectRecord): RuntimeDirectTransportSnapshot {
    return {
      streamId: record.streamId,
      sourceNodeId: record.grant.sourceNodeId,
      destinationNodeId: record.grant.destinationNodeId,
      targetPort: record.grant.targetPort,
      mode: "direct",
      state: record.endedAt !== null
        ? "closed"
        : record.committed
          ? "active"
          : "negotiating",
      bytesTx: record.bytesTx,
      bytesRx: record.bytesRx,
      createdAt: record.createdAt,
      connectedAt: record.connectedAt,
      endedAt: record.endedAt,
      connectRttMs: record.connectRttMs,
    };
  }

  private validateGrantForRole(
    streamId: string,
    grant: DirectSessionGrant,
    role: "source" | "destination",
  ): void {
    if (
      grant.protocol !== DIRECT_TRANSPORT_PROTOCOL
      || grant.connectionId !== streamId
      || grant.expiresAt <= Date.now()
      || (role === "source" ? grant.sourceNodeId : grant.destinationNodeId) !== this.nodeId
    ) {
      throw new Error("direct_grant_identity_is_invalid");
    }
  }

  private rejectOffer(streamId: string, connectionId: string, reason: string): void {
    this.callbacks.send("runtime.direct.fallback", {
      streamId,
      connectionId,
      reason: sanitizeReason(reason),
    });
  }

  private rejectSource(streamId: string, connectionId: string, reason: string): void {
    const error = new Error(sanitizeReason(reason));
    this.callbacks.send("runtime.direct.fallback", {
      streamId,
      connectionId,
      reason: error.message,
    });
    this.callbacks.onSourceFailed(streamId, error);
  }
}

function directCandidates(
  port: number,
  configuredHosts: string[] | undefined,
  listenHost: string | undefined,
): DirectTransportCandidate[] {
  const candidates: DirectTransportCandidate[] = [];
  for (const host of configuredHosts ?? []) {
    addCandidate(candidates, validateHost(host), port, "configured");
  }
  const bound = listenHost?.trim();
  if (bound && bound !== "0.0.0.0" && bound !== "::") {
    addCandidate(candidates, validateHost(bound), port, "configured");
  }
  for (const addresses of Object.values(networkInterfaces())) {
    for (const address of addresses ?? []) {
      if (
        address.internal
        || address.family !== "IPv4"
        || !isLanIpv4(address.address)
      ) continue;
      addCandidate(candidates, address.address, port, "lan");
    }
  }
  return candidates.slice(0, MAX_CANDIDATES);
}

/**
 * A same-segment private address avoids NAT and is cheapest. On a WAN source,
 * the verified public mapping is tried before unreachable private addresses.
 * Configured endpoints remain an explicit operator override.
 */
export function orderDirectCandidates(
  candidates: DirectTransportCandidate[],
  localHosts?: string[],
): DirectTransportCandidate[] {
  const localSubnets = localHosts
    ? localHosts.flatMap((host) => ipv4Subnet(host, "255.255.255.0"))
    : localPrivateIpv4Subnets();
  return candidates
    .map((candidate, index) => ({ candidate, index }))
    .sort((left, right) =>
      directCandidateRank(left.candidate, localSubnets)
      - directCandidateRank(right.candidate, localSubnets)
      || left.index - right.index
    )
    .map(({ candidate }) => candidate);
}

function directCandidateRank(
  candidate: DirectTransportCandidate,
  localSubnets: Ipv4Subnet[],
): number {
  if (
    isPrivateIpv4(candidate.host)
    && localSubnets.some((subnet) => subnetContains(subnet, candidate.host))
  ) return 0;
  if (candidate.scope === "configured") return 1;
  if (candidate.scope === "public-mapped") return 2;
  return 3;
}

interface Ipv4Subnet {
  address: number;
  mask: number;
}

function localPrivateIpv4Subnets(): Ipv4Subnet[] {
  return Object.values(networkInterfaces()).flatMap((addresses) =>
    (addresses ?? [])
      .filter((address) =>
        address.family === "IPv4"
        && !address.internal
        && isPrivateIpv4(address.address)
      )
      .flatMap((address) => ipv4Subnet(address.address, address.netmask))
  );
}

function ipv4Subnet(address: string, netmask: string): Ipv4Subnet[] {
  const addressNumber = ipv4Number(address);
  const maskNumber = ipv4Number(netmask);
  return addressNumber === null || maskNumber === null
    ? []
    : [{ address: addressNumber, mask: maskNumber }];
}

function subnetContains(subnet: Ipv4Subnet, host: string): boolean {
  const candidate = ipv4Number(host);
  return candidate !== null
    && (candidate & subnet.mask) === (subnet.address & subnet.mask);
}

function ipv4Number(host: string): number | null {
  const parts = host.split(".").map(Number);
  if (
    parts.length !== 4
    || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)
  ) return null;
  return (
    ((parts[0]! << 24) >>> 0)
    + (parts[1]! << 16)
    + (parts[2]! << 8)
    + parts[3]!
  ) >>> 0;
}

function isLanIpv4(host: string): boolean {
  const octets = host.split(".").map(Number);
  if (
    octets.length !== 4
    || octets.some((part) => !Number.isInteger(part) || part < 0 || part > 255)
  ) return false;
  const [a, b] = octets as [number, number, number, number];
  return (
    a === 10
    || (a === 172 && b >= 16 && b <= 31)
    || (a === 192 && b === 168)
    || (a === 169 && b === 254)
    || (a === 100 && b >= 64 && b <= 127)
  );
}

function addCandidate(
  candidates: DirectTransportCandidate[],
  host: string,
  port: number,
  scope: DirectTransportCandidate["scope"],
): void {
  if (candidates.some((candidate) => candidate.host === host && candidate.port === port)) return;
  candidates.push({ host, port: boundedPort(port), scope });
}

function validateHost(value: string): string {
  if (
    typeof value !== "string"
    || value.length < 1
    || value.length > 253
    || /[\u0000-\u0020/?#@]/.test(value)
  ) {
    throw new Error("direct_candidate_host_is_invalid");
  }
  return value;
}

function validateCandidateHost(candidate: DirectTransportCandidate): string {
  const host = validateHost(candidate.host);
  if (candidate.scope === "public-mapped" && !isPublicIpv4(host)) {
    throw new Error("direct_public_candidate_is_invalid");
  }
  if (candidate.scope === "lan" && !isLanIpv4(host)) {
    throw new Error("direct_lan_candidate_is_invalid");
  }
  return host;
}

function boundedPort(value: number): number {
  return boundedInteger(value, 1, 65_535, "direct_candidate_port_is_invalid");
}

function boundedInteger(value: number, minimum: number, maximum: number, code: string): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(code);
  return value;
}

function checkedBytes(current: number, added: number, maximum: number): number {
  const next = current + added;
  if (!Number.isSafeInteger(next) || next > maximum) {
    throw new Error("direct_session_byte_limit_exceeded");
  }
  return next;
}

function sanitizeReason(value: unknown): string {
  const text = value instanceof Error ? value.message : String(value);
  return text.replace(/[^\w:.-]/g, "_").slice(0, 256) || "direct_transport_failed";
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(error: Error): void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((accept, fail) => {
    resolve = accept;
    reject = fail;
  });
  return { promise, resolve, reject };
}

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  code: string,
): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(code)), timeoutMs);
        timer.unref();
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
