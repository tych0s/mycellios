import { describe, expect, it } from "vitest";
import { parseAutoDistributionConfig } from "../src/distribution/auto-distribute.js";
import { evaluateTwoHostPreflight } from "../src/distribution/two-host-preflight.js";

describe("two-host preflight", () => {
  it("accepts a configured two-host route without claiming unmeasured evidence", () => {
    const report = evaluateTwoHostPreflight(config(), { python312: true, configuredEnvironmentNames: new Set(["REMOTE_TOKEN"]),
      preparedStageRanges: [{ nodeId: "host-a", startLayer: 0, endLayer: 12 }, { nodeId: "host-b", startLayer: 12, endLayer: 24 }] });
    expect(report.ready).toBe(true);
    expect(report.capabilities.measuredTransport).toBe(false);
    expect(report.diagnostics).toContainEqual(expect.objectContaining({ code: "transport_unmeasured", severity: "warning" }));
  });

  it("rejects loopback, duplicate identity and missing authentication", () => {
    const parsed = config();
    parsed.nodes[1]!.id = "host-a";
    parsed.nodes[1]!.endpoint.host = "127.0.0.1";
    const report = evaluateTwoHostPreflight(parsed, { python312: true, configuredEnvironmentNames: new Set() });
    expect(report.ready).toBe(false);
    expect(report.diagnostics.map((entry) => entry.code)).toEqual(expect.arrayContaining(["node_identity_duplicate", "stage_endpoint_loopback", "remote_auth_missing"]));
  });

  it("marks documentation addresses as dry-run only", () => {
    const value = raw(); value.nodes[1]!.endpoint.host = "192.0.2.11";
    const report = evaluateTwoHostPreflight(parseAutoDistributionConfig(value), { python312: true, configuredEnvironmentNames: new Set(["REMOTE_TOKEN"]) });
    expect(report).toMatchObject({ ready: false, dryRun: true });
    expect(report.diagnostics).toContainEqual(expect.objectContaining({ code: "stage_endpoint_placeholder" }));
  });
});

function config() { return parseAutoDistributionConfig(raw()); }
function raw(): any {
  return { schema: "gdlp-auto-distribute/1", model: { source: "example/model", revision: "immutable-revision", publicName: "physical" },
    nodes: [node("host-a", "192.168.1.10", { kind: "local" }), node("host-b", "192.168.1.11", { kind: "http", endpoint: "http://192.168.1.11:9750", authTokenEnv: "REMOTE_TOKEN", requestTimeoutMs: 30000 })],
    links: [link("host-a", "host-b"), link("host-b", "host-a")], distribution: { minimumStages: 2, maximumStages: 2, allowLossyActivation: false },
    workload: { promptTokens: 64, outputTokens: 16, contextTokens: 2048, concurrentSequences: 1, minRouteAvailability: 0.9, batchWindowMs: 0, p95: true },
    runtime: { pythonExecutable: "python", stagePythonExecutable: "python", pythonPath: "python", hfHome: "runtime/hf-cache", apiEndpoint: { host: "127.0.0.1", port: 8081 }, apiAdvertiseHost: "192.168.1.10", returnEndpoint: { host: "192.168.1.10", port: 18100 }, returnBindHost: "0.0.0.0", threadsPerStage: 1, connectTimeoutSeconds: 300, readinessTimeoutMs: 600000, maxOutputTokens: 256 },
    canary: { prompt: "OK", maxTokens: 16, timeoutMs: 300000 }, artifactsDirectory: "runtime/physical" };
}
function node(id: string, host: string, agent: any) { return { id, region: "lan", endpoint: { host, port: 18101 }, memoryMiB: 8192, reserveMiB: 1024, decodeScale: 1, prefillScale: 1, codecScale: 1, powerWatts: 65, availability: 0.99, agent }; }
function link(from: string, to: string) { return { from, to, oneWayLatencyMs: 1, jitterP95Ms: 0.5, bandwidthMbps: 1000, lossRate: 0, availability: 0.99 }; }
