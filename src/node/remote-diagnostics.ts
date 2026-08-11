import { createHash } from "node:crypto";
import {
  redactDiagnosticDetails,
  redactDiagnosticText,
  type RemoteDiagnosticEvent,
} from "../contracts/remote-diagnostics.js";
import type { SystemLogEntry, SystemLogSnapshot } from "../contracts/control-api.js";

const UPLOAD_INTERVAL_MS = 30_000;
const UPLOAD_DEBOUNCE_MS = 1_000;

export interface RemoteDiagnosticsMetadata {
  sourceId: string;
  appVersion: string;
  platform: string;
  arch: string;
}

export interface RemoteDiagnosticsUploaderOptions {
  coordinatorUrl: string;
  networkToken: string;
  metadata: RemoteDiagnosticsMetadata;
  readLogs: () => SystemLogSnapshot;
  fetchImpl?: typeof fetch;
}

export class RemoteDiagnosticsUploader {
  private readonly fetchImpl: typeof fetch;
  private interval: NodeJS.Timeout | null = null;
  private debounce: NodeJS.Timeout | null = null;
  private active: Promise<void> | null = null;
  private lastBatchDigest = "";
  private stopped = false;

  constructor(private readonly options: RemoteDiagnosticsUploaderOptions) {
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  start(): void {
    if (this.interval || this.stopped) return;
    this.interval = setInterval(
      () => void this.flush().catch(() => undefined),
      UPLOAD_INTERVAL_MS,
    );
    this.interval.unref();
    this.notify();
  }

  notify(): void {
    if (this.stopped || this.debounce) return;
    this.debounce = setTimeout(() => {
      this.debounce = null;
      void this.flush().catch(() => undefined);
    }, UPLOAD_DEBOUNCE_MS);
    this.debounce.unref();
  }

  flush(): Promise<void> {
    if (this.stopped) return Promise.resolve();
    if (this.active) return this.active;
    this.active = this.uploadLatest().finally(() => {
      this.active = null;
    });
    return this.active;
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.interval) clearInterval(this.interval);
    if (this.debounce) clearTimeout(this.debounce);
    this.interval = null;
    this.debounce = null;
    await this.active?.catch(() => undefined);
  }

  private async uploadLatest(): Promise<void> {
    const events = remoteDiagnosticEvents(
      this.options.readLogs(),
      this.options.metadata,
    ).slice(0, 100);
    if (events.length === 0) return;
    const batchDigest = createHash("sha256")
      .update(events.map((event) => event.id).join("\n"))
      .digest("hex");
    if (batchDigest === this.lastBatchDigest) return;
    const response = await this.fetchImpl(
      new URL("/internal/v1/diagnostics", this.options.coordinatorUrl),
      {
        method: "POST",
        signal: AbortSignal.timeout(8_000),
        headers: {
          authorization: `Bearer ${this.options.networkToken}`,
          "content-type": "application/json",
        },
        body: JSON.stringify({ events }),
      },
    );
    if (!response.ok) throw new Error(`remote_diagnostics_http_${response.status}`);
    this.lastBatchDigest = batchDigest;
  }
}

export function remoteDiagnosticEvents(
  snapshot: SystemLogSnapshot,
  metadata: RemoteDiagnosticsMetadata,
): RemoteDiagnosticEvent[] {
  return snapshot.entries.map((entry) => remoteDiagnosticEvent(entry, metadata));
}

function remoteDiagnosticEvent(
  entry: SystemLogEntry,
  metadata: RemoteDiagnosticsMetadata,
): RemoteDiagnosticEvent {
  const safeMessage = redactDiagnosticText(entry.message).slice(0, 1_200) || entry.event;
  const safeDetails = entry.details
    ? redactDiagnosticDetails(entry.details).slice(0, 2_000)
    : undefined;
  const digest = createHash("sha256")
    .update([
      metadata.sourceId,
      entry.at,
      entry.event,
      safeMessage,
      safeDetails ?? "",
    ].join("\0"))
    .digest("hex");
  return {
    id: `diag_${digest}`,
    ...metadata,
    level: entry.level,
    source: entry.source,
    event: entry.event,
    message: safeMessage,
    ...(safeDetails ? { details: safeDetails } : {}),
    occurredAt: validIsoDate(entry.at),
  };
}

function validIsoDate(value: string): string {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp)
    ? new Date(timestamp).toISOString()
    : new Date().toISOString();
}
