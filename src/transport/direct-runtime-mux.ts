import { EventEmitter } from "node:events";
import type { DirectSecureChannel } from "./direct-secure-channel.js";

export const DIRECT_RUNTIME_MUX_PROTOCOL = "mycellios-direct-runtime-mux/1" as const;

const MAGIC = 0x4d594431;
const HEADER_BYTES = 24;
const DEFAULT_RECEIVE_WINDOW_BYTES = 4 * 1024 * 1024;
const DEFAULT_MAX_CHUNK_BYTES = 48 * 1024;
const DEFAULT_MAX_STREAMS = 256;
const DEFAULT_MAX_PENDING_WRITE_BYTES = 64 * 1024 * 1024;
const MAX_STREAM_ID_BYTES = 256;
const MAX_METADATA_BYTES = 8 * 1024;
const MAX_RESET_REASON_BYTES = 1_024;

const enum FrameType {
  Open = 1,
  Data = 2,
  Acknowledge = 3,
  Close = 4,
  Reset = 5,
}

interface RuntimeFrame {
  type: FrameType;
  streamId: string;
  offset: number;
  creditBytes: number;
  payload: Buffer;
}

interface PendingWrite {
  data: Buffer;
  offset: number;
  resolve(): void;
  reject(error: Error): void;
}

interface IncomingChunk {
  offset: number;
  data: Buffer;
}

interface StreamState {
  id: string;
  stream: DirectRuntimeStream;
  localOpened: boolean;
  remoteOpened: boolean;
  localClosed: boolean;
  remoteClosed: boolean;
  sendOffset: number;
  acknowledgedOffset: number;
  receiveOffset: number;
  sendCreditBytes: number;
  receiveBufferedBytes: number;
  pendingWriteBytes: number;
  pendingWrites: PendingWrite[];
  incoming: IncomingChunk[];
  drainingIncoming: boolean;
  handler: DirectRuntimeDataHandler | null;
}

export interface DirectRuntimeMuxOptions {
  receiveWindowBytes?: number;
  maximumChunkBytes?: number;
  maximumStreams?: number;
  maximumPendingWriteBytes?: number;
}

export type DirectRuntimeDataHandler = (
  data: Buffer,
  offset: number,
) => void | Promise<void>;

interface MuxEvents {
  stream: [DirectRuntimeStream];
  error: [Error];
  close: [];
}

interface StreamEvents {
  end: [];
  reset: [Error];
  error: [Error];
}

export class DirectRuntimeStream extends EventEmitter<StreamEvents> {
  constructor(
    readonly id: string,
    readonly metadata: Buffer,
    private readonly mux: DirectRuntimeMux,
  ) {
    super();
    this.on("error", () => undefined);
  }

  write(data: Uint8Array): Promise<void> {
    return this.mux.write(this.id, data);
  }

  setDataHandler(handler: DirectRuntimeDataHandler): void {
    this.mux.setDataHandler(this.id, handler);
  }

  close(): Promise<void> {
    return this.mux.closeStream(this.id);
  }

  reset(reason: string): Promise<void> {
    return this.mux.resetStream(this.id, reason);
  }

  snapshot(): {
    sendOffset: number;
    acknowledgedOffset: number;
    receiveOffset: number;
    sendCreditBytes: number;
    receiveBufferedBytes: number;
    pendingWriteBytes: number;
  } {
    return this.mux.streamSnapshot(this.id);
  }
}

/**
 * Multiplexes resumable byte streams over one mutually authenticated channel.
 *
 * Credits are per stream, so a slow stage cannot consume unbounded memory or
 * block unrelated streams. ACK, close and reset frames use a separate control
 * queue that always drains before data frames.
 */
export class DirectRuntimeMux extends EventEmitter<MuxEvents> {
  private readonly streams = new Map<string, StreamState>();
  private readonly receiveWindowBytes: number;
  private readonly maximumChunkBytes: number;
  private readonly maximumStreams: number;
  private readonly maximumPendingWriteBytes: number;
  private readonly controlFrames: Buffer[] = [];
  private readonly dataFrames = new Map<string, Buffer[]>();
  private readonly dataRoundRobin: string[] = [];
  private pumping = false;
  private closed = false;

  constructor(
    private readonly channel: DirectSecureChannel,
    options: DirectRuntimeMuxOptions = {},
  ) {
    super();
    this.receiveWindowBytes = boundedInteger(
      options.receiveWindowBytes ?? DEFAULT_RECEIVE_WINDOW_BYTES,
      1_024,
      64 * 1024 * 1024,
      "direct_mux_receive_window_is_invalid",
    );
    this.maximumChunkBytes = boundedInteger(
      options.maximumChunkBytes ?? DEFAULT_MAX_CHUNK_BYTES,
      256,
      60 * 1024,
      "direct_mux_chunk_limit_is_invalid",
    );
    this.maximumStreams = boundedInteger(
      options.maximumStreams ?? DEFAULT_MAX_STREAMS,
      1,
      4_096,
      "direct_mux_stream_limit_is_invalid",
    );
    this.maximumPendingWriteBytes = boundedInteger(
      options.maximumPendingWriteBytes ?? DEFAULT_MAX_PENDING_WRITE_BYTES,
      1_024,
      64 * 1024 * 1024,
      "direct_mux_pending_write_limit_is_invalid",
    );
    this.on("error", () => undefined);
    channel.on("data", (value: Buffer) => this.receiveFrame(value));
    channel.on("error", (error: Error) => this.fail(error));
    channel.on("close", () => this.fail(new Error("direct_mux_channel_closed")));
  }

  async openStream(
    streamId: string,
    metadata: Uint8Array = Buffer.alloc(0),
  ): Promise<DirectRuntimeStream> {
    this.ensureOpen();
    const id = validateStreamId(streamId);
    if (this.streams.has(id)) throw new Error("direct_mux_stream_is_duplicate");
    if (this.streams.size >= this.maximumStreams) {
      throw new Error("direct_mux_stream_capacity_exceeded");
    }
    const document = Buffer.from(metadata);
    if (document.byteLength > MAX_METADATA_BYTES) {
      throw new Error("direct_mux_metadata_is_too_large");
    }
    const stream = new DirectRuntimeStream(id, document, this);
    const state = this.createState(stream, true, false);
    this.streams.set(id, state);
    await this.queueFrame({
      type: FrameType.Open,
      streamId: id,
      offset: 0,
      creditBytes: this.receiveWindowBytes,
      payload: document,
    }, true);
    return stream;
  }

  async write(streamId: string, value: Uint8Array): Promise<void> {
    this.ensureOpen();
    const state = this.requiredStream(streamId);
    if (state.localClosed) throw new Error("direct_mux_stream_is_closed");
    if (value.byteLength < 1) throw new Error("direct_mux_write_is_empty");
    const projectedPendingBytes = Math.max(
      0,
      state.pendingWriteBytes + value.byteLength - state.sendCreditBytes,
    );
    if (projectedPendingBytes > this.maximumPendingWriteBytes) {
      throw new Error("direct_mux_pending_write_capacity_exceeded");
    }
    const data = Buffer.from(value);
    const promises: Promise<void>[] = [];
    for (let offset = 0; offset < data.byteLength; offset += this.maximumChunkBytes) {
      const piece = Buffer.from(data.subarray(
        offset,
        Math.min(data.byteLength, offset + this.maximumChunkBytes),
      ));
      promises.push(new Promise<void>((resolve, reject) => {
        state.pendingWrites.push({
          data: piece,
          offset: state.sendOffset,
          resolve,
          reject,
        });
        state.pendingWriteBytes += piece.byteLength;
        state.sendOffset += piece.byteLength;
      }));
    }
    this.flushWrites(state);
    await Promise.all(promises);
  }

  setDataHandler(streamId: string, handler: DirectRuntimeDataHandler): void {
    const state = this.requiredStream(streamId);
    state.handler = handler;
    this.scheduleIncomingDrain(state);
  }

  async closeStream(streamId: string): Promise<void> {
    const state = this.requiredStream(streamId);
    if (state.localClosed) return;
    state.localClosed = true;
    await this.queueFrame({
      type: FrameType.Close,
      streamId: state.id,
      offset: state.sendOffset,
      creditBytes: 0,
      payload: Buffer.alloc(0),
    }, true);
    this.collectStream(state);
  }

  async resetStream(streamId: string, reason: string): Promise<void> {
    const state = this.requiredStream(streamId);
    const payload = Buffer.from(
      boundedText(reason, MAX_RESET_REASON_BYTES, "direct_mux_reset_reason_is_invalid"),
      "utf8",
    );
    await this.queueFrame({
      type: FrameType.Reset,
      streamId: state.id,
      offset: state.receiveOffset,
      creditBytes: 0,
      payload,
    }, true);
    this.failStream(state, new Error(reason));
  }

  streamSnapshot(streamId: string): {
    sendOffset: number;
    acknowledgedOffset: number;
    receiveOffset: number;
    sendCreditBytes: number;
    receiveBufferedBytes: number;
    pendingWriteBytes: number;
  } {
    const state = this.requiredStream(streamId);
    return {
      sendOffset: state.sendOffset,
      acknowledgedOffset: state.acknowledgedOffset,
      receiveOffset: state.receiveOffset,
      sendCreditBytes: state.sendCreditBytes,
      receiveBufferedBytes: state.receiveBufferedBytes,
      pendingWriteBytes: state.pendingWriteBytes,
    };
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const error = new Error("direct_mux_closed");
    for (const state of this.streams.values()) this.failStream(state, error);
    this.streams.clear();
    this.controlFrames.length = 0;
    this.dataFrames.clear();
    this.dataRoundRobin.length = 0;
    await this.channel.close();
    this.emit("close");
  }

  private receiveFrame(value: Buffer): void {
    if (this.closed) return;
    let frame: RuntimeFrame;
    try {
      frame = decodeFrame(value, this.maximumChunkBytes);
      this.handleFrame(frame);
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private handleFrame(frame: RuntimeFrame): void {
    switch (frame.type) {
      case FrameType.Open:
        this.receiveOpen(frame);
        return;
      case FrameType.Data:
        this.receiveData(frame);
        return;
      case FrameType.Acknowledge:
        this.receiveAcknowledgement(frame);
        return;
      case FrameType.Close:
        this.receiveClose(frame);
        return;
      case FrameType.Reset:
        this.receiveReset(frame);
        return;
    }
  }

  private receiveOpen(frame: RuntimeFrame): void {
    if (frame.offset !== 0 || frame.creditBytes < 1) {
      throw new Error("direct_mux_open_frame_is_invalid");
    }
    if (this.streams.has(frame.streamId)) {
      throw new Error("direct_mux_stream_is_duplicate");
    }
    if (this.streams.size >= this.maximumStreams) {
      void this.queueFrame({
        type: FrameType.Reset,
        streamId: frame.streamId,
        offset: 0,
        creditBytes: 0,
        payload: Buffer.from("direct_mux_stream_capacity_exceeded"),
      }, true);
      return;
    }
    const stream = new DirectRuntimeStream(frame.streamId, frame.payload, this);
    const state = this.createState(stream, false, true);
    state.sendCreditBytes = frame.creditBytes;
    this.streams.set(state.id, state);
    void this.queueFrame({
      type: FrameType.Acknowledge,
      streamId: state.id,
      offset: 0,
      creditBytes: this.receiveWindowBytes,
      payload: Buffer.alloc(0),
    }, true);
    this.emit("stream", stream);
  }

  private receiveData(frame: RuntimeFrame): void {
    if (frame.creditBytes !== 0 || frame.payload.byteLength < 1) {
      throw new Error("direct_mux_data_frame_is_invalid");
    }
    const state = this.requiredStream(frame.streamId);
    if (state.remoteClosed) throw new Error("direct_mux_data_after_close");
    const endOffset = frame.offset + frame.payload.byteLength;
    if (frame.offset < state.receiveOffset) {
      if (endOffset <= state.receiveOffset) {
        void this.queueAcknowledgement(state, 0);
        return;
      }
      throw new Error("direct_mux_overlapping_replay");
    }
    const queuedEnd = state.receiveOffset + state.receiveBufferedBytes;
    if (frame.offset !== queuedEnd) throw new Error("direct_mux_receive_offset_mismatch");
    if (state.receiveBufferedBytes + frame.payload.byteLength > this.receiveWindowBytes) {
      throw new Error("direct_mux_receive_window_exceeded");
    }
    state.receiveBufferedBytes += frame.payload.byteLength;
    state.incoming.push({ offset: frame.offset, data: frame.payload });
    this.scheduleIncomingDrain(state);
  }

  private receiveAcknowledgement(frame: RuntimeFrame): void {
    if (frame.payload.byteLength !== 0 || frame.creditBytes < 0) {
      throw new Error("direct_mux_ack_frame_is_invalid");
    }
    const state = this.requiredStream(frame.streamId);
    if (
      frame.offset < state.acknowledgedOffset
      || frame.offset > state.sendOffset
    ) {
      throw new Error("direct_mux_ack_offset_is_invalid");
    }
    if (state.localOpened && !state.remoteOpened && frame.offset === 0) {
      state.remoteOpened = true;
    }
    state.acknowledgedOffset = frame.offset;
    state.sendCreditBytes += frame.creditBytes;
    if (state.sendCreditBytes > this.receiveWindowBytes) {
      throw new Error("direct_mux_credit_overflow");
    }
    this.flushWrites(state);
  }

  private receiveClose(frame: RuntimeFrame): void {
    if (frame.creditBytes !== 0 || frame.payload.byteLength !== 0) {
      throw new Error("direct_mux_close_frame_is_invalid");
    }
    const state = this.requiredStream(frame.streamId);
    if (frame.offset !== state.receiveOffset + state.receiveBufferedBytes) {
      throw new Error("direct_mux_close_offset_is_invalid");
    }
    state.remoteClosed = true;
    if (state.receiveBufferedBytes === 0) {
      state.stream.emit("end");
      this.collectStream(state);
    }
  }

  private receiveReset(frame: RuntimeFrame): void {
    const state = this.streams.get(frame.streamId);
    if (!state) return;
    const reason = frame.payload.toString("utf8") || "direct_mux_remote_reset";
    this.failStream(state, new Error(reason));
  }

  private async drainIncoming(state: StreamState): Promise<void> {
    if (state.drainingIncoming || !state.handler) return;
    state.drainingIncoming = true;
    try {
      while (state.incoming.length > 0) {
        const chunk = state.incoming.shift()!;
        try {
          await state.handler(chunk.data, chunk.offset);
        } catch (error) {
          const failure = error instanceof Error ? error : new Error(String(error));
          await this.resetStream(state.id, failure.message);
          return;
        }
        if (this.closed || !this.streams.has(state.id)) return;
        state.receiveOffset += chunk.data.byteLength;
        state.receiveBufferedBytes -= chunk.data.byteLength;
        await this.queueAcknowledgement(state, chunk.data.byteLength);
      }
      if (state.remoteClosed && state.receiveBufferedBytes === 0) {
        state.stream.emit("end");
        this.collectStream(state);
      }
    } finally {
      state.drainingIncoming = false;
    }
  }

  private scheduleIncomingDrain(state: StreamState): void {
    void this.drainIncoming(state).catch((error: unknown) => {
      if (this.closed || !this.streams.has(state.id)) return;
      this.fail(error instanceof Error ? error : new Error(String(error)));
    });
  }

  private flushWrites(state: StreamState): void {
    if (this.closed || !state.remoteOpened) return;
    while (state.pendingWrites.length > 0) {
      const next = state.pendingWrites[0]!;
      if (next.data.byteLength > state.sendCreditBytes) return;
      state.pendingWrites.shift();
      state.pendingWriteBytes -= next.data.byteLength;
      state.sendCreditBytes -= next.data.byteLength;
      void this.queueFrame({
        type: FrameType.Data,
        streamId: state.id,
        offset: next.offset,
        creditBytes: 0,
        payload: next.data,
      }, false).then(next.resolve, next.reject);
    }
  }

  private queueAcknowledgement(state: StreamState, creditBytes: number): Promise<void> {
    return this.queueFrame({
      type: FrameType.Acknowledge,
      streamId: state.id,
      offset: state.receiveOffset,
      creditBytes,
      payload: Buffer.alloc(0),
    }, true);
  }

  private queueFrame(frame: RuntimeFrame, control: boolean): Promise<void> {
    if (this.closed) return Promise.reject(new Error("direct_mux_is_closed"));
    const encoded = encodeFrame(frame, this.maximumChunkBytes);
    if (control) {
      this.controlFrames.push(encoded);
    } else {
      const queue = this.dataFrames.get(frame.streamId) ?? [];
      if (!this.dataFrames.has(frame.streamId)) {
        this.dataFrames.set(frame.streamId, queue);
        this.dataRoundRobin.push(frame.streamId);
      }
      queue.push(encoded);
    }
    return this.pump();
  }

  private async pump(): Promise<void> {
    if (this.pumping || this.closed) return;
    this.pumping = true;
    try {
      while (!this.closed) {
        const frame = this.controlFrames.shift() ?? this.nextDataFrame();
        if (!frame) break;
        await this.channel.send(frame);
      }
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally {
      this.pumping = false;
      if (!this.closed && (this.controlFrames.length > 0 || this.dataFrames.size > 0)) {
        void this.pump();
      }
    }
  }

  private nextDataFrame(): Buffer | null {
    while (this.dataRoundRobin.length > 0) {
      const streamId = this.dataRoundRobin.shift()!;
      const queue = this.dataFrames.get(streamId);
      const frame = queue?.shift();
      if (queue && queue.length > 0) {
        this.dataRoundRobin.push(streamId);
      } else {
        this.dataFrames.delete(streamId);
      }
      if (frame) return frame;
    }
    return null;
  }

  private createState(
    stream: DirectRuntimeStream,
    localOpened: boolean,
    remoteOpened: boolean,
  ): StreamState {
    return {
      id: stream.id,
      stream,
      localOpened,
      remoteOpened,
      localClosed: false,
      remoteClosed: false,
      sendOffset: 0,
      acknowledgedOffset: 0,
      receiveOffset: 0,
      sendCreditBytes: 0,
      receiveBufferedBytes: 0,
      pendingWriteBytes: 0,
      pendingWrites: [],
      incoming: [],
      drainingIncoming: false,
      handler: null,
    };
  }

  private requiredStream(streamId: string): StreamState {
    const state = this.streams.get(validateStreamId(streamId));
    if (!state) throw new Error("direct_mux_stream_not_found");
    return state;
  }

  private collectStream(state: StreamState): void {
    if (
      state.localClosed
      && state.remoteClosed
      && state.pendingWrites.length === 0
      && state.receiveBufferedBytes === 0
    ) {
      this.streams.delete(state.id);
      this.dataFrames.delete(state.id);
      const index = this.dataRoundRobin.indexOf(state.id);
      if (index >= 0) this.dataRoundRobin.splice(index, 1);
    }
  }

  private failStream(state: StreamState, error: Error): void {
    if (!this.streams.delete(state.id)) return;
    state.localClosed = true;
    state.remoteClosed = true;
    for (const write of state.pendingWrites.splice(0)) write.reject(error);
    state.pendingWriteBytes = 0;
    state.incoming.length = 0;
    state.receiveBufferedBytes = 0;
    this.dataFrames.delete(state.id);
    const index = this.dataRoundRobin.indexOf(state.id);
    if (index >= 0) this.dataRoundRobin.splice(index, 1);
    state.stream.emit("reset", error);
    state.stream.emit("error", error);
  }

  private fail(error: Error): void {
    if (this.closed) return;
    this.closed = true;
    for (const state of [...this.streams.values()]) this.failStream(state, error);
    this.emit("error", error);
    this.emit("close");
  }

  private ensureOpen(): void {
    if (this.closed) throw new Error("direct_mux_is_closed");
  }
}

function encodeFrame(frame: RuntimeFrame, maximumChunkBytes: number): Buffer {
  const streamId = Buffer.from(validateStreamId(frame.streamId), "utf8");
  if (streamId.byteLength > MAX_STREAM_ID_BYTES) {
    throw new Error("direct_mux_stream_id_is_too_large");
  }
  if (!Number.isSafeInteger(frame.offset) || frame.offset < 0) {
    throw new Error("direct_mux_frame_offset_is_invalid");
  }
  if (
    !Number.isSafeInteger(frame.creditBytes)
    || frame.creditBytes < 0
    || frame.creditBytes > 64 * 1024 * 1024
  ) {
    throw new Error("direct_mux_frame_credit_is_invalid");
  }
  const maximumPayload = frame.type === FrameType.Data
    ? maximumChunkBytes
    : frame.type === FrameType.Open
      ? MAX_METADATA_BYTES
      : frame.type === FrameType.Reset
        ? MAX_RESET_REASON_BYTES
        : 0;
  if (frame.payload.byteLength > maximumPayload) {
    throw new Error("direct_mux_frame_payload_is_invalid");
  }
  const header = Buffer.allocUnsafe(HEADER_BYTES);
  header.writeUInt32BE(MAGIC, 0);
  header.writeUInt8(frame.type, 4);
  header.writeUInt8(0, 5);
  header.writeUInt16BE(streamId.byteLength, 6);
  header.writeBigUInt64BE(BigInt(frame.offset), 8);
  header.writeUInt32BE(frame.creditBytes, 16);
  header.writeUInt32BE(frame.payload.byteLength, 20);
  return Buffer.concat([header, streamId, frame.payload]);
}

function decodeFrame(value: Buffer, maximumChunkBytes: number): RuntimeFrame {
  if (value.byteLength < HEADER_BYTES) throw new Error("direct_mux_frame_is_truncated");
  if (value.readUInt32BE(0) !== MAGIC) throw new Error("direct_mux_frame_magic_is_invalid");
  const type = value.readUInt8(4);
  if (type < FrameType.Open || type > FrameType.Reset || value.readUInt8(5) !== 0) {
    throw new Error("direct_mux_frame_type_is_invalid");
  }
  const streamIdBytes = value.readUInt16BE(6);
  const offset = Number(value.readBigUInt64BE(8));
  const creditBytes = value.readUInt32BE(16);
  const payloadBytes = value.readUInt32BE(20);
  if (
    streamIdBytes < 1
    || streamIdBytes > MAX_STREAM_ID_BYTES
    || !Number.isSafeInteger(offset)
    || HEADER_BYTES + streamIdBytes + payloadBytes !== value.byteLength
  ) {
    throw new Error("direct_mux_frame_shape_is_invalid");
  }
  const streamId = validateStreamId(
    value.subarray(HEADER_BYTES, HEADER_BYTES + streamIdBytes).toString("utf8"),
  );
  const payload = Buffer.from(value.subarray(HEADER_BYTES + streamIdBytes));
  const frame = {
    type: type as FrameType,
    streamId,
    offset,
    creditBytes,
    payload,
  };
  // Reuse the encoder's per-type limits and strict field validation.
  encodeFrame(frame, maximumChunkBytes);
  return frame;
}

function validateStreamId(value: string): string {
  return boundedText(value, MAX_STREAM_ID_BYTES, "direct_mux_stream_id_is_invalid");
}

function boundedText(value: unknown, maximumBytes: number, code: string): string {
  if (typeof value !== "string") throw new Error(code);
  const normalized = value.trim();
  if (
    !normalized
    || Buffer.byteLength(normalized, "utf8") > maximumBytes
    || /[\u0000-\u001f\u007f]/.test(normalized)
  ) {
    throw new Error(code);
  }
  return normalized;
}

function boundedInteger(
  value: number,
  minimum: number,
  maximum: number,
  code: string,
): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new Error(code);
  }
  return value;
}
