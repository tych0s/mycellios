import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LiveNetworkTelemetry } from "./Panel";

describe("LiveNetworkTelemetry", () => {
  it("uses only current online capacity and active job states", () => {
    const snapshot = {
      capturedAt: new Date().toISOString(),
      version: "0.73.1",
      buildIdentity: null,
      summary: { registered: 2, connected: 1, online: 1, mobile: 0, offeredVramMb: 196_608, completedJobs: 1 },
      workers: [{
        id: "online-peer",
        kind: "desktop",
        status: "online",
        connected: true,
        region: "eu-west",
        offeredVramMb: 65_536,
        reliability: 1,
        jobsCompleted: 1,
        lastSeenAt: new Date().toISOString(),
        gpus: [{
          id: "gpu-0",
          vendor: "NVIDIA",
          model: "GPU",
          physicalVramMb: 65_536,
          offeredVramMb: 65_536,
          freeOfferedVramMb: 32_768,
        }],
        deployments: [],
        quarantined: false,
        quarantineExpiresAt: null,
      }, {
        id: "offline-peer",
        kind: "cell",
        status: "offline",
        connected: false,
        region: "eu-west",
        offeredVramMb: 131_072,
        reliability: 0.9,
        jobsCompleted: 0,
        lastSeenAt: new Date(0).toISOString(),
        gpus: [],
        deployments: [],
        quarantined: false,
        quarantineExpiresAt: null,
      }],
      models: [
        { id: "model-a", replicas: 2, pipelines: 1 },
        { id: "model-b", replicas: 1, pipelines: 0 },
      ],
      requestedModels: [],
      jobs: [
        { id: "queued", model: "model-a", status: "queued", workerId: null, inputTokens: 0, outputTokens: 0, failureCode: null, createdAt: "", updatedAt: "" },
        { id: "running", model: "model-a", status: "running", workerId: "online-peer", inputTokens: 0, outputTokens: 0, failureCode: null, createdAt: "", updatedAt: "" },
        { id: "streaming", model: "model-a", status: "streaming", workerId: "online-peer", inputTokens: 0, outputTokens: 1, failureCode: null, createdAt: "", updatedAt: "" },
        { id: "completed", model: "model-a", status: "completed", workerId: "online-peer", inputTokens: 1, outputTokens: 1, failureCode: null, createdAt: "", updatedAt: "" },
        { id: "failed", model: "model-a", status: "failed", workerId: null, inputTokens: 0, outputTokens: 0, failureCode: "no_capacity", createdAt: "", updatedAt: "" },
      ],
    } satisfies Parameters<typeof LiveNetworkTelemetry>[0]["snapshot"];

    const html = renderToStaticMarkup(<LiveNetworkTelemetry snapshot={snapshot} />);

    expect(html).toContain('aria-label="Current live network data"');
    expect(html).toContain('data-period="current"');
    expect(html).toContain('data-evidence="coordinator-snapshot"');
    expect(html).toContain("LIVE");
    expect(html).toContain("3 replicas · 1 pipelines");
    expect(html).toContain("<strong>3</strong>");
    expect(html).toContain("2 running · 1 queued");
    expect(html.match(/data-kpi=/g)).toHaveLength(4);
    expect(html).not.toContain("Mesh VRAM");
  });
});
