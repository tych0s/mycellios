import { afterEach, describe, expect, it } from "vitest";
import {
  redactDiagnosticDetails,
  remoteDiagnosticBatchSchema,
  type RemoteDiagnosticEvent,
} from "../src/contracts/remote-diagnostics.js";
import {
  RemoteDiagnosticsUploader,
  remoteDiagnosticEvents,
} from "../src/desktop/remote-diagnostics.js";
import { MeshDatabase } from "../src/storage/database.js";
import { MeshStore } from "../src/storage/store.js";

const databases: MeshDatabase[] = [];

afterEach(() => {
  for (const database of databases.splice(0)) database.close();
});

describe("remote desktop diagnostics", () => {
  it("redacts secrets before producing a stable upload event", () => {
    const snapshot = {
      capturedAt: "2026-07-29T12:00:00.000Z",
      truncated: false,
      source: "desktop-file" as const,
      entries: [{
        id: "local-entry",
        at: "2026-07-29T11:59:00.000Z",
        level: "error" as const,
        source: "runtime" as const,
        event: "runtime-start-failed",
        message: "Bearer visible-secret token=also-visible",
        details: JSON.stringify({
          error: "request failed?api_key=visible-key",
          password: "visible-password",
          nested: { authorization: "Bearer another-secret" },
        }),
      }],
    };
    const metadata = {
      sourceId: "3d7e5bea-c718-4e82-8f05-dfbda14257fc",
      appVersion: "0.2.54",
      platform: "win32",
      arch: "x64",
    };

    const first = remoteDiagnosticEvents(snapshot, metadata);
    const second = remoteDiagnosticEvents(snapshot, metadata);

    expect(first).toEqual(second);
    expect(first[0]?.id).toMatch(/^diag_[a-f0-9]{64}$/);
    expect(JSON.stringify(first)).not.toContain("visible-secret");
    expect(JSON.stringify(first)).not.toContain("visible-key");
    expect(JSON.stringify(first)).not.toContain("visible-password");
    expect(remoteDiagnosticBatchSchema.parse({ events: first })).toBeTruthy();
  });

  it("redacts sensitive keys inside structured details", () => {
    const redacted = redactDiagnosticDetails(JSON.stringify({
      cookie: "session",
      safe: "kept",
      nested: { apiKey: "private", reason: "timeout" },
    }));
    expect(JSON.parse(redacted)).toEqual({
      cookie: "[REDACTED]",
      safe: "kept",
      nested: { apiKey: "[REDACTED]", reason: "timeout" },
    });
  });

  it("uploads the latest sanitized batch once with the network credential", async () => {
    const requests: Array<{ url: string; authorization: string | null; body: string }> = [];
    const uploader = new RemoteDiagnosticsUploader({
      coordinatorUrl: "https://coordinator.example.test",
      networkToken: "network-secret",
      metadata: {
        sourceId: "3d7e5bea-c718-4e82-8f05-dfbda14257fc",
        appVersion: "0.2.54",
        platform: "win32",
        arch: "x64",
      },
      readLogs: () => ({
        capturedAt: "2026-07-29T12:00:00.000Z",
        truncated: false,
        source: "desktop-file",
        entries: [{
          id: "entry",
          at: "2026-07-29T12:00:00.000Z",
          level: "error",
          source: "runtime",
          event: "runtime-start-failed",
          message: "token=visible",
        }],
      }),
      fetchImpl: async (input, init) => {
        const headers = new Headers(init?.headers);
        requests.push({
          url: String(input),
          authorization: headers.get("authorization"),
          body: String(init?.body),
        });
        return new Response("{}", { status: 201 });
      },
    });

    await uploader.flush();
    await uploader.flush();
    await uploader.stop();

    expect(requests).toHaveLength(1);
    expect(requests[0]).toMatchObject({
      url: "https://coordinator.example.test/internal/v1/diagnostics",
      authorization: "Bearer network-secret",
    });
    expect(requests[0]?.body).not.toContain("visible");
  });

  it("deduplicates, filters and prunes expired events", () => {
    const database = new MeshDatabase(":memory:");
    databases.push(database);
    const store = new MeshStore(database);
    const recent = event({
      id: `diag_${"a".repeat(64)}`,
      level: "error",
      occurredAt: new Date().toISOString(),
    });
    const expired = event({
      id: `diag_${"b".repeat(64)}`,
      level: "info",
      occurredAt: "2026-01-01T00:00:00.000Z",
    });

    expect(store.appendDiagnosticEvents([recent, expired])).toEqual({
      accepted: 2,
      duplicates: 0,
    });
    expect(store.appendDiagnosticEvents([recent])).toEqual({
      accepted: 0,
      duplicates: 1,
    });
    expect(store.listDiagnosticEvents()).toHaveLength(1);
    expect(store.listDiagnosticEvents({ level: "error" })[0]).toMatchObject({
      id: recent.id,
      message: recent.message,
    });
    expect(store.listDiagnosticEvents({ level: "warning" })).toEqual([]);
  });
});

function event(
  overrides: Partial<RemoteDiagnosticEvent>,
): RemoteDiagnosticEvent {
  return { ...baseEvent(), ...overrides };
}

function baseEvent(): RemoteDiagnosticEvent {
  return {
    id: `diag_${"c".repeat(64)}`,
    sourceId: "3d7e5bea-c718-4e82-8f05-dfbda14257fc",
    appVersion: "0.2.54",
    platform: "win32",
    arch: "x64",
    level: "error" as const,
    source: "runtime" as const,
    event: "runtime-start-failed",
    message: "CUDA runtime could not start.",
    occurredAt: "2026-07-29T12:00:00.000Z",
  };
}
