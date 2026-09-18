import { createHmac, timingSafeEqual } from "node:crypto";
import { constants } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import { createConnection, type NetConnectOpts } from "node:net";
import { dirname, join } from "node:path";
import {
  CHECKPOINT_CONTROL_ENDPOINT_NAME,
  CHECKPOINT_CONTROL_SCHEMA,
  CHECKPOINT_CONTROL_SOCKET_NAME,
  MAX_CHECKPOINT_CONTROL_HEADER_BYTES as MAXIMUM_HEADER_BYTES,
} from "../contracts/activation-checkpoint-control.js";
import { MAX_ACTIVATION_CHECKPOINT_BYTES } from "../contracts/activation-checkpoint.js";

export interface ActivationCheckpointControlOptions {
  workspacePath: string;
  token: string;
  signal?: AbortSignal;
  platform?: NodeJS.Platform;
}

/** A signed endpoint binds the per-process secret to the actual loopback port. */
async function controlEndpoint(
  options: ActivationCheckpointControlOptions,
): Promise<{ connection: NetConnectOpts; token?: string }> {
  const descriptorPath = join(options.workspacePath, CHECKPOINT_CONTROL_ENDPOINT_NAME);
  let metadata;
  try {
    metadata = await lstat(descriptorPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    if ((options.platform ?? process.platform) === "win32") {
      throw new Error("activation_checkpoint_control_endpoint_is_unavailable");
    }
    return { connection: { path: join(options.workspacePath, CHECKPOINT_CONTROL_SOCKET_NAME) } };
  }
  if (!metadata.isFile() || metadata.isSymbolicLink() || metadata.size > MAXIMUM_HEADER_BYTES) {
    throw new Error("activation_checkpoint_control_endpoint_is_invalid");
  }
  const actualPath = await realpath(descriptorPath);
  if (dirname(actualPath) !== await realpath(options.workspacePath)) {
    throw new Error("activation_checkpoint_control_endpoint_is_invalid");
  }
  const handle = await open(descriptorPath, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  let text: string;
  try {
    const opened = await handle.stat();
    if (!opened.isFile() || opened.size > MAXIMUM_HEADER_BYTES
      || opened.dev !== metadata.dev || opened.ino !== metadata.ino) {
      throw new Error("activation_checkpoint_control_endpoint_is_invalid");
    }
    const buffer = Buffer.alloc(MAXIMUM_HEADER_BYTES + 1);
    const { bytesRead } = await handle.read(buffer, 0, buffer.byteLength, 0);
    if (bytesRead > MAXIMUM_HEADER_BYTES) {
      throw new Error("activation_checkpoint_control_endpoint_is_invalid");
    }
    text = buffer.subarray(0, bytesRead).toString("utf8");
  } finally {
    await handle.close();
  }
  let descriptor: Record<string, unknown>;
  try {
    descriptor = JSON.parse(text) as Record<string, unknown>;
  } catch {
    throw new Error("activation_checkpoint_control_endpoint_is_invalid");
  }
  if (!descriptor || Array.isArray(descriptor)
    || Object.keys(descriptor).sort().join(",") !== "host,port,schema,signature"
    || descriptor.schema !== CHECKPOINT_CONTROL_SCHEMA || descriptor.host !== "127.0.0.1"
    || !Number.isSafeInteger(descriptor.port) || (descriptor.port as number) < 1
    || (descriptor.port as number) > 65_535
    || typeof descriptor.signature !== "string" || !/^[a-f0-9]{64}$/u.test(descriptor.signature)
    || !/^[a-f0-9]{64}$/u.test(options.token)) {
    throw new Error("activation_checkpoint_control_endpoint_is_invalid");
  }
  const signature = createHmac("sha256", Buffer.from(options.token, "hex"))
    .update(`${CHECKPOINT_CONTROL_SCHEMA}\n127.0.0.1\n${descriptor.port}`).digest();
  if (!timingSafeEqual(signature, Buffer.from(descriptor.signature, "hex"))) {
    throw new Error("activation_checkpoint_control_endpoint_authentication_failed");
  }
  return {
    connection: { host: "127.0.0.1", port: descriptor.port as number, family: 4 },
    token: options.token,
  };
}

export async function activationCheckpointControlRequest(
  options: ActivationCheckpointControlOptions,
  header: Record<string, unknown>,
  payload: Buffer | undefined,
  maximumResponseBytes: number,
): Promise<{ payload: Buffer; committedPosition?: number }> {
  if (!Number.isSafeInteger(maximumResponseBytes) || maximumResponseBytes < 0
    || maximumResponseBytes > MAX_ACTIVATION_CHECKPOINT_BYTES) {
    throw new Error("activation_checkpoint_response_limit_is_invalid");
  }
  if (!Number.isSafeInteger(header.requestId) || (header.requestId as number) < 0
    || !Number.isSafeInteger(header.maxBytes) || (header.maxBytes as number) < 1
    || (header.maxBytes as number) > MAX_ACTIVATION_CHECKPOINT_BYTES
    || (payload !== undefined && payload.byteLength > (header.maxBytes as number))) {
    throw new Error("activation_checkpoint_control_request_is_invalid");
  }
  if ((header.operation !== "capture" && header.operation !== "restore")
    || (header.operation === "capture" && payload !== undefined)
    || (header.operation === "restore" && (payload === undefined
      || payload.byteLength !== header.payloadBytes
      || !Number.isSafeInteger(header.committedPosition) || (header.committedPosition as number) < 1))) {
    throw new Error("activation_checkpoint_control_request_is_invalid");
  }
  const abortError = () => new Error("activation_checkpoint_control_aborted");
  if (options.signal?.aborted) throw abortError();
  const endpoint = await controlEndpoint(options);
  if (options.signal?.aborted) throw abortError();
  const request = `${JSON.stringify({ ...header, ...(endpoint.token ? { token: endpoint.token } : {}) })}\n`;
  if (Buffer.byteLength(request) > MAXIMUM_HEADER_BYTES) {
    throw new Error("activation_checkpoint_control_request_header_is_too_large");
  }
  return new Promise((resolveRequest, rejectRequest) => {
    const socket = createConnection(endpoint.connection);
    let headerBytes = Buffer.alloc(0);
    let response: { payloadBytes: number; committedPosition?: number } | undefined;
    const chunks: Buffer[] = [];
    let received = 0;
    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      options.signal?.removeEventListener("abort", abort);
      socket.destroy();
      if (error) rejectRequest(error);
      else resolveRequest({
        payload: Buffer.concat(chunks, received),
        ...(response?.committedPosition === undefined
          ? {} : { committedPosition: response.committedPosition }),
      });
    };
    const abort = () => finish(abortError());
    const timeout = setTimeout(() => finish(new Error("activation_checkpoint_control_timed_out")), 35_000);
    timeout.unref();
    socket.once("error", (error) => finish(error));
    options.signal?.addEventListener("abort", abort, { once: true });
    if (options.signal?.aborted) { abort(); return; }
    socket.once("connect", () => {
      if (settled) return;
      socket.write(request);
      if (payload) socket.write(payload);
    });
    socket.on("data", (chunk: Buffer) => {
      if (settled) return;
      let body = chunk;
      if (response === undefined) {
        const newline = chunk.indexOf(0x0a);
        const part = newline < 0 ? chunk : chunk.subarray(0, newline);
        if (headerBytes.byteLength + part.byteLength > MAXIMUM_HEADER_BYTES) {
          finish(new Error("activation_checkpoint_control_header_is_too_large"));
          return;
        }
        headerBytes = Buffer.concat([headerBytes, part]);
        if (newline < 0) return;
        let parsed: unknown;
        try { parsed = JSON.parse(headerBytes.toString("utf8")); } catch {
          finish(new Error("activation_checkpoint_control_header_is_invalid"));
          return;
        }
        if (!parsed || typeof parsed !== "object" || !("ok" in parsed)) {
          finish(new Error("activation_checkpoint_control_response_is_invalid"));
          return;
        }
        const record = parsed as Record<string, unknown>;
        if (record.ok !== true) {
          const detail = typeof record.error === "string"
            ? record.error.replaceAll(options.token, "[redacted]").slice(0, 256) : "";
          finish(new Error(`activation_checkpoint_control_failed${detail ? `:${detail}` : ""}`));
          return;
        }
        if (!Number.isSafeInteger(record.payloadBytes) || (record.payloadBytes as number) < 0
          || (record.payloadBytes as number) > maximumResponseBytes) {
          finish(new Error("activation_checkpoint_control_payload_size_is_invalid"));
          return;
        }
        response = { payloadBytes: record.payloadBytes as number,
          ...(record.committedPosition === undefined ? {} : { committedPosition: record.committedPosition as number }),
        };
        body = chunk.subarray(newline + 1);
      }
      received += body.byteLength;
      if (received > response.payloadBytes) {
        finish(new Error("activation_checkpoint_control_payload_has_trailing_bytes"));
        return;
      }
      if (body.byteLength) chunks.push(Buffer.from(body));
      if (received === response.payloadBytes) finish();
    });
    socket.once("close", () => {
      if (!settled) finish(new Error("activation_checkpoint_control_ended_early"));
    });
    socket.once("end", () => {
      if (!settled) finish(new Error("activation_checkpoint_control_ended_early"));
    });
  });
}
