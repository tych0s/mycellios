import {
  createCipheriv,
  createDecipheriv,
  createHmac,
  hkdfSync,
  randomBytes,
  timingSafeEqual,
} from "node:crypto";
import { EventEmitter, once } from "node:events";
import { createServer, Socket, type Server } from "node:net";

export const DIRECT_TRANSPORT_PROTOCOL = "mycellios-direct/1" as const;

const HANDSHAKE_MAX_BYTES = 4_096;
const DEFAULT_MAX_FRAME_BYTES = 64 * 1024;
const DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000;
const FRAME_HEADER_BYTES = 12;
const AUTH_TAG_BYTES = 16;

export interface DirectSessionGrant {
  protocol: typeof DIRECT_TRANSPORT_PROTOCOL;
  connectionId: string;
  sourceNodeId: string;
  destinationNodeId: string;
  targetPort: number;
  expiresAt: number;
  secret: string;
}

export interface DirectSecureServerOptions {
  host?: string;
  port?: number;
  resolveGrant(connectionId: string): DirectSessionGrant | null;
  onChannel(channel: DirectSecureChannel, grant: DirectSessionGrant): void;
  onRejected?(error: Error): void;
  now?(): number;
  maximumFrameBytes?: number;
  handshakeTimeoutMs?: number;
}

export interface DirectSecureClientOptions {
  host: string;
  port: number;
  grant: DirectSessionGrant;
  timeoutMs?: number;
  maximumFrameBytes?: number;
  now?(): number;
}

interface ClientHello {
  protocol: typeof DIRECT_TRANSPORT_PROTOCOL;
  connectionId: string;
  sourceNodeId: string;
  destinationNodeId: string;
  targetPort: number;
  expiresAt: number;
  clientNonce: string;
  mac: string;
}

interface ServerHello {
  protocol: typeof DIRECT_TRANSPORT_PROTOCOL;
  connectionId: string;
  serverNonce: string;
  mac: string;
}

export function createDirectSessionGrant(input: {
  connectionId: string;
  sourceNodeId: string;
  destinationNodeId: string;
  targetPort: number;
  expiresAt: number;
  secret?: Buffer;
}): DirectSessionGrant {
  const connectionId = identifier(input.connectionId, "direct_connection_id_is_invalid");
  const sourceNodeId = identifier(input.sourceNodeId, "direct_source_node_id_is_invalid");
  const destinationNodeId = identifier(
    input.destinationNodeId,
    "direct_destination_node_id_is_invalid",
  );
  if (sourceNodeId === destinationNodeId) {
    throw new Error("direct_transport_requires_distinct_nodes");
  }
  const targetPort = boundedPort(input.targetPort);
  if (!Number.isSafeInteger(input.expiresAt) || input.expiresAt <= 0) {
    throw new Error("direct_transport_expiry_is_invalid");
  }
  const secret = input.secret ?? randomBytes(32);
  if (secret.byteLength !== 32) throw new Error("direct_transport_secret_is_invalid");
  return {
    protocol: DIRECT_TRANSPORT_PROTOCOL,
    connectionId,
    sourceNodeId,
    destinationNodeId,
    targetPort,
    expiresAt: input.expiresAt,
    secret: secret.toString("base64url"),
  };
}

export class DirectSecureServer {
  private server: Server | null = null;
  private readonly consumedConnectionIds = new Map<string, number>();
  private readonly sockets = new Set<Socket>();

  constructor(private readonly options: DirectSecureServerOptions) {}

  async listen(): Promise<{ host: string; port: number }> {
    if (this.server) throw new Error("direct_transport_server_already_listening");
    const server = createServer((socket) => {
      this.sockets.add(socket);
      socket.once("close", () => this.sockets.delete(socket));
      void this.accept(socket).catch((error: unknown) => {
        const failure = error instanceof Error ? error : new Error(String(error));
        this.options.onRejected?.(failure);
        socket.destroy();
      });
    });
    this.server = server;
    const host = this.options.host ?? "127.0.0.1";
    const port = boundedPort(this.options.port ?? 0);
    server.listen({ host, port });
    await once(server, "listening");
    const address = server.address();
    if (!address || typeof address === "string") {
      await this.close();
      throw new Error("direct_transport_server_address_is_invalid");
    }
    return { host: address.address, port: address.port };
  }

  async close(): Promise<void> {
    const server = this.server;
    this.server = null;
    for (const socket of this.sockets) socket.destroy();
    this.sockets.clear();
    if (!server) return;
    const closed = once(server, "close").catch(() => undefined);
    server.close();
    await closed;
  }

  private async accept(socket: Socket): Promise<void> {
    socket.setNoDelay(true);
    const timeoutMs = boundedPositiveInteger(
      this.options.handshakeTimeoutMs,
      DEFAULT_HANDSHAKE_TIMEOUT_MS,
      250,
      60_000,
    );
    const reader = new SocketReader(socket);
    const hello = parseClientHello(
      await withTimeout(readDocument(reader), timeoutMs, "direct_handshake_timeout"),
    );
    const grant = this.options.resolveGrant(hello.connectionId);
    if (!grant) throw new Error("direct_transport_grant_not_found");
    const now = this.options.now?.() ?? Date.now();
    validateGrant(grant, now);
    for (const [connectionId, expiresAt] of this.consumedConnectionIds) {
      if (expiresAt <= now) this.consumedConnectionIds.delete(connectionId);
    }
    if (
      grant.sourceNodeId !== hello.sourceNodeId
      || grant.destinationNodeId !== hello.destinationNodeId
      || grant.targetPort !== hello.targetPort
      || grant.expiresAt !== hello.expiresAt
    ) {
      throw new Error("direct_transport_grant_identity_mismatch");
    }
    if (this.consumedConnectionIds.has(grant.connectionId)) {
      throw new Error("direct_transport_grant_was_already_consumed");
    }
    const secret = grantSecret(grant);
    const clientNonce = base64Bytes(hello.clientNonce, 32, "direct_client_nonce_is_invalid");
    verifyMac(
      secret,
      clientHelloTranscript(hello),
      hello.mac,
      "direct_client_authentication_failed",
    );
    const serverNonce = randomBytes(32);
    const response: ServerHello = {
      protocol: DIRECT_TRANSPORT_PROTOCOL,
      connectionId: grant.connectionId,
      serverNonce: serverNonce.toString("base64url"),
      mac: mac(
        secret,
        serverHelloTranscript(grant, hello.clientNonce, serverNonce.toString("base64url")),
      ),
    };
    await writeDocument(socket, response);
    this.consumedConnectionIds.set(grant.connectionId, grant.expiresAt);
    const keys = deriveDirectionalKeys(grant, secret, clientNonce, serverNonce);
    const channel = new DirectSecureChannel(
      socket,
      reader,
      keys.serverWriteKey,
      keys.clientWriteKey,
      grant.connectionId,
      boundedPositiveInteger(
        this.options.maximumFrameBytes,
        DEFAULT_MAX_FRAME_BYTES,
        1_024,
        8 * 1024 * 1024,
      ),
    );
    this.options.onChannel(channel, grant);
    channel.start();
  }
}

export class DirectSecureChannel extends EventEmitter {
  private writeSequence = 0n;
  private readSequence = 0n;
  private started = false;
  private closed = false;

  constructor(
    private readonly socket: Socket,
    private readonly reader: SocketReader,
    private readonly writeKey: Buffer,
    private readonly readKey: Buffer,
    private readonly connectionId: string,
    private readonly maximumFrameBytes: number,
  ) {
    super();
    // A rejected channel may have no consumer yet. Preserve fail-closed
    // teardown without turning a transport fault into an unhandled exception.
    this.on("error", () => undefined);
  }

  static async connect(options: DirectSecureClientOptions): Promise<DirectSecureChannel> {
    const now = options.now?.() ?? Date.now();
    validateGrant(options.grant, now);
    const socket = new Socket();
    socket.setNoDelay(true);
    const timeoutMs = boundedPositiveInteger(
      options.timeoutMs,
      DEFAULT_HANDSHAKE_TIMEOUT_MS,
      250,
      60_000,
    );
    const connectPromise = new Promise<void>((resolve, reject) => {
      socket.once("connect", resolve);
      socket.once("error", reject);
      socket.connect({ host: options.host, port: boundedPort(options.port) });
    });
    await withTimeout(connectPromise, timeoutMs, "direct_connect_timeout").catch((error) => {
      socket.destroy();
      throw error;
    });
    const reader = new SocketReader(socket);
    const secret = grantSecret(options.grant);
    const clientNonce = randomBytes(32);
    const helloWithoutMac = {
      protocol: DIRECT_TRANSPORT_PROTOCOL,
      connectionId: options.grant.connectionId,
      sourceNodeId: options.grant.sourceNodeId,
      destinationNodeId: options.grant.destinationNodeId,
      targetPort: options.grant.targetPort,
      expiresAt: options.grant.expiresAt,
      clientNonce: clientNonce.toString("base64url"),
    };
    const hello: ClientHello = {
      ...helloWithoutMac,
      mac: mac(secret, clientHelloTranscript(helloWithoutMac)),
    };
    await writeDocument(socket, hello);
    const response = parseServerHello(
      await withTimeout(readDocument(reader), timeoutMs, "direct_handshake_timeout"),
    );
    if (
      response.connectionId !== options.grant.connectionId
      || response.protocol !== DIRECT_TRANSPORT_PROTOCOL
    ) {
      socket.destroy();
      throw new Error("direct_server_identity_mismatch");
    }
    const serverNonce = base64Bytes(
      response.serverNonce,
      32,
      "direct_server_nonce_is_invalid",
    );
    verifyMac(
      secret,
      serverHelloTranscript(options.grant, hello.clientNonce, response.serverNonce),
      response.mac,
      "direct_server_authentication_failed",
    );
    const keys = deriveDirectionalKeys(
      options.grant,
      secret,
      clientNonce,
      serverNonce,
    );
    const channel = new DirectSecureChannel(
      socket,
      reader,
      keys.clientWriteKey,
      keys.serverWriteKey,
      options.grant.connectionId,
      boundedPositiveInteger(
        options.maximumFrameBytes,
        DEFAULT_MAX_FRAME_BYTES,
        1_024,
        8 * 1024 * 1024,
      ),
    );
    channel.start();
    return channel;
  }

  start(): void {
    if (this.started) return;
    this.started = true;
    void this.receiveLoop().catch((error: unknown) => {
      if (this.closed) return;
      this.emit("error", error instanceof Error ? error : new Error(String(error)));
      this.destroy();
    });
  }

  async send(data: Uint8Array): Promise<void> {
    if (this.closed || this.socket.destroyed) throw new Error("direct_channel_is_closed");
    if (data.byteLength < 1 || data.byteLength > this.maximumFrameBytes) {
      throw new Error("direct_frame_size_is_invalid");
    }
    const header = Buffer.allocUnsafe(FRAME_HEADER_BYTES);
    header.writeBigUInt64BE(this.writeSequence, 0);
    header.writeUInt32BE(data.byteLength, 8);
    const nonce = frameNonce(this.writeSequence);
    const cipher = createCipheriv("aes-256-gcm", this.writeKey, nonce);
    cipher.setAAD(frameAad(this.connectionId, header));
    const encrypted = Buffer.concat([cipher.update(data), cipher.final()]);
    const frame = Buffer.concat([header, encrypted, cipher.getAuthTag()]);
    this.writeSequence += 1n;
    if (!this.socket.write(frame)) await once(this.socket, "drain");
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.socket.end();
    if (!this.socket.destroyed) {
      await Promise.race([
        once(this.socket, "close"),
        new Promise((resolve) => setTimeout(resolve, 1_000)),
      ]);
    }
    this.socket.destroy();
    this.emit("close");
  }

  destroy(): void {
    if (this.closed) return;
    this.closed = true;
    this.socket.destroy();
    this.emit("close");
  }

  private async receiveLoop(): Promise<void> {
    while (!this.closed) {
      const header = await this.reader.readExactly(FRAME_HEADER_BYTES);
      const sequence = header.readBigUInt64BE(0);
      const length = header.readUInt32BE(8);
      if (sequence !== this.readSequence) {
        throw new Error("direct_frame_sequence_is_invalid");
      }
      if (length < 1 || length > this.maximumFrameBytes) {
        throw new Error("direct_frame_size_is_invalid");
      }
      const body = await this.reader.readExactly(length + AUTH_TAG_BYTES);
      const encrypted = body.subarray(0, length);
      const tag = body.subarray(length);
      const decipher = createDecipheriv(
        "aes-256-gcm",
        this.readKey,
        frameNonce(sequence),
      );
      decipher.setAAD(frameAad(this.connectionId, header));
      decipher.setAuthTag(tag);
      let plaintext: Buffer;
      try {
        plaintext = Buffer.concat([decipher.update(encrypted), decipher.final()]);
      } catch {
        throw new Error("direct_frame_authentication_failed");
      }
      this.readSequence += 1n;
      this.emit("data", plaintext);
    }
  }
}

class SocketReader {
  private buffer = Buffer.alloc(0);
  private endedError: Error | null = null;
  private waiter: (() => void) | null = null;

  constructor(socket: Socket) {
    socket.on("data", (chunk: Buffer) => {
      this.buffer = this.buffer.length === 0
        ? Buffer.from(chunk)
        : Buffer.concat([this.buffer, chunk]);
      this.wake();
    });
    socket.once("end", () => {
      this.endedError = new Error("direct_transport_socket_ended");
      this.wake();
    });
    socket.once("error", (error) => {
      this.endedError = error;
      this.wake();
    });
    socket.once("close", () => {
      this.endedError ??= new Error("direct_transport_socket_closed");
      this.wake();
    });
  }

  async readExactly(length: number): Promise<Buffer> {
    while (this.buffer.length < length) {
      if (this.endedError) throw this.endedError;
      await new Promise<void>((resolve) => {
        this.waiter = resolve;
      });
    }
    const value = this.buffer.subarray(0, length);
    this.buffer = this.buffer.subarray(length);
    return value;
  }

  private wake(): void {
    const waiter = this.waiter;
    this.waiter = null;
    waiter?.();
  }
}

async function readDocument(reader: SocketReader): Promise<unknown> {
  const lengthBuffer = await reader.readExactly(4);
  const length = lengthBuffer.readUInt32BE(0);
  if (length < 2 || length > HANDSHAKE_MAX_BYTES) {
    throw new Error("direct_handshake_document_size_is_invalid");
  }
  const document = await reader.readExactly(length);
  try {
    return JSON.parse(document.toString("utf8")) as unknown;
  } catch {
    throw new Error("direct_handshake_document_is_invalid");
  }
}

async function writeDocument(socket: Socket, value: unknown): Promise<void> {
  const document = Buffer.from(JSON.stringify(value), "utf8");
  if (document.byteLength > HANDSHAKE_MAX_BYTES) {
    throw new Error("direct_handshake_document_size_is_invalid");
  }
  const length = Buffer.allocUnsafe(4);
  length.writeUInt32BE(document.byteLength, 0);
  if (!socket.write(Buffer.concat([length, document]))) await once(socket, "drain");
}

function parseClientHello(value: unknown): ClientHello {
  const record = strictRecord(value, [
    "protocol",
    "connectionId",
    "sourceNodeId",
    "destinationNodeId",
    "targetPort",
    "expiresAt",
    "clientNonce",
    "mac",
  ], "direct_client_hello_is_invalid");
  if (
    record.protocol !== DIRECT_TRANSPORT_PROTOCOL
    || !Number.isSafeInteger(record.targetPort)
    || !Number.isSafeInteger(record.expiresAt)
  ) {
    throw new Error("direct_client_hello_is_invalid");
  }
  return {
    protocol: DIRECT_TRANSPORT_PROTOCOL,
    connectionId: identifier(record.connectionId, "direct_client_hello_is_invalid"),
    sourceNodeId: identifier(record.sourceNodeId, "direct_client_hello_is_invalid"),
    destinationNodeId: identifier(
      record.destinationNodeId,
      "direct_client_hello_is_invalid",
    ),
    targetPort: boundedPort(record.targetPort as number),
    expiresAt: record.expiresAt as number,
    clientNonce: text(record.clientNonce, 128, "direct_client_hello_is_invalid"),
    mac: text(record.mac, 128, "direct_client_hello_is_invalid"),
  };
}

function parseServerHello(value: unknown): ServerHello {
  const record = strictRecord(value, [
    "protocol",
    "connectionId",
    "serverNonce",
    "mac",
  ], "direct_server_hello_is_invalid");
  if (record.protocol !== DIRECT_TRANSPORT_PROTOCOL) {
    throw new Error("direct_server_hello_is_invalid");
  }
  return {
    protocol: DIRECT_TRANSPORT_PROTOCOL,
    connectionId: identifier(record.connectionId, "direct_server_hello_is_invalid"),
    serverNonce: text(record.serverNonce, 128, "direct_server_hello_is_invalid"),
    mac: text(record.mac, 128, "direct_server_hello_is_invalid"),
  };
}

function validateGrant(grant: DirectSessionGrant, now: number): void {
  if (grant.protocol !== DIRECT_TRANSPORT_PROTOCOL) {
    throw new Error("direct_transport_grant_protocol_is_invalid");
  }
  identifier(grant.connectionId, "direct_connection_id_is_invalid");
  identifier(grant.sourceNodeId, "direct_source_node_id_is_invalid");
  identifier(grant.destinationNodeId, "direct_destination_node_id_is_invalid");
  boundedPort(grant.targetPort);
  grantSecret(grant);
  if (!Number.isSafeInteger(grant.expiresAt) || grant.expiresAt <= now) {
    throw new Error("direct_transport_grant_is_expired");
  }
}

function deriveDirectionalKeys(
  grant: DirectSessionGrant,
  secret: Buffer,
  clientNonce: Buffer,
  serverNonce: Buffer,
): { clientWriteKey: Buffer; serverWriteKey: Buffer } {
  const salt = Buffer.concat([clientNonce, serverNonce]);
  const info = Buffer.from(
    `${DIRECT_TRANSPORT_PROTOCOL}\0${grant.connectionId}\0${grant.sourceNodeId}\0${grant.destinationNodeId}\0${grant.targetPort}`,
    "utf8",
  );
  const material = Buffer.from(hkdfSync("sha256", secret, salt, info, 64));
  return {
    clientWriteKey: material.subarray(0, 32),
    serverWriteKey: material.subarray(32, 64),
  };
}

function clientHelloTranscript(
  hello: Omit<ClientHello, "mac"> | ClientHello,
): Buffer {
  return Buffer.from([
    "client",
    hello.protocol,
    hello.connectionId,
    hello.sourceNodeId,
    hello.destinationNodeId,
    String(hello.targetPort),
    String(hello.expiresAt),
    hello.clientNonce,
  ].join("\0"), "utf8");
}

function serverHelloTranscript(
  grant: DirectSessionGrant,
  clientNonce: string,
  serverNonce: string,
): Buffer {
  return Buffer.from([
    "server",
    grant.protocol,
    grant.connectionId,
    grant.sourceNodeId,
    grant.destinationNodeId,
    String(grant.targetPort),
    String(grant.expiresAt),
    clientNonce,
    serverNonce,
  ].join("\0"), "utf8");
}

function frameAad(connectionId: string, header: Buffer): Buffer {
  return Buffer.concat([
    Buffer.from(`${DIRECT_TRANSPORT_PROTOCOL}\0${connectionId}\0`, "utf8"),
    header,
  ]);
}

function frameNonce(sequence: bigint): Buffer {
  const nonce = Buffer.alloc(12);
  nonce.writeUInt32BE(0x4d594331, 0);
  nonce.writeBigUInt64BE(sequence, 4);
  return nonce;
}

function mac(secret: Buffer, transcript: Buffer): string {
  return createHmac("sha256", secret).update(transcript).digest("base64url");
}

function verifyMac(
  secret: Buffer,
  transcript: Buffer,
  encoded: string,
  code: string,
): void {
  const expected = createHmac("sha256", secret).update(transcript).digest();
  const actual = base64Bytes(encoded, expected.byteLength, code);
  if (!timingSafeEqual(expected, actual)) throw new Error(code);
}

function grantSecret(grant: DirectSessionGrant): Buffer {
  return base64Bytes(grant.secret, 32, "direct_transport_secret_is_invalid");
}

function base64Bytes(value: string, length: number, code: string): Buffer {
  if (typeof value !== "string" || !/^[A-Za-z0-9_-]+$/.test(value)) {
    throw new Error(code);
  }
  const bytes = Buffer.from(value, "base64url");
  if (bytes.byteLength !== length || bytes.toString("base64url") !== value) {
    throw new Error(code);
  }
  return bytes;
}

function strictRecord(
  value: unknown,
  keys: readonly string[],
  code: string,
): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(code);
  const record = value as Record<string, unknown>;
  const actual = Object.keys(record).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) {
    throw new Error(code);
  }
  return record;
}

function identifier(value: unknown, code: string): string {
  return text(value, 192, code);
}

function text(value: unknown, maximum: number, code: string): string {
  if (typeof value !== "string") throw new Error(code);
  const normalized = value.trim();
  if (!normalized || normalized.length > maximum || /[\u0000-\u001f\u007f]/.test(normalized)) {
    throw new Error(code);
  }
  return normalized;
}

function boundedPort(value: number): number {
  if (!Number.isInteger(value) || value < 0 || value > 65_535) {
    throw new Error("direct_transport_port_is_invalid");
  }
  return value;
}

function boundedPositiveInteger(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
): number {
  const candidate = value ?? fallback;
  if (!Number.isInteger(candidate) || candidate < minimum || candidate > maximum) {
    throw new Error("direct_transport_limit_is_invalid");
  }
  return candidate;
}

async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  code: string,
): Promise<T> {
  let timer: NodeJS.Timeout | null = null;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error(code)), timeoutMs);
        timer.unref();
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
