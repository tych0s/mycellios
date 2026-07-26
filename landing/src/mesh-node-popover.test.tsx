import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { MeshNodePopover } from "./Panel";

describe("MeshNodePopover", () => {
  it("shows verified peer details and clamps the horizontal map position", () => {
    const worker = {
      id: "peer-gx10-3c9a",
      kind: "desktop",
      status: "online",
      connected: true,
      region: "eu-west",
      offeredVramMb: 131_072,
      reliability: 0.987,
      jobsCompleted: 42,
      lastSeenAt: new Date().toISOString(),
      agentVersion: "0.73.1",
      gpus: [{
        id: "gpu-0",
        vendor: "NVIDIA",
        model: "GX10-3C9A",
        physicalVramMb: 131_072,
        offeredVramMb: 131_072,
        freeOfferedVramMb: 89_000,
      }],
      deployments: [{
        deploymentId: "deployment-1",
        model: "singulared/Ornith-1.0-35B-MTP-GGUF:Q8_0",
        mode: "openai-compatible",
        freeSlots: 1,
        tokensPerSecond: 12.5,
      }],
    } satisfies Parameters<typeof MeshNodePopover>[0]["worker"];

    const html = renderToStaticMarkup(
      <MeshNodePopover worker={worker} position={{ x: 95, y: 72 }} sharePercent={32.4} onClose={vi.fn()} />,
    );

    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-label="Node details for GX10-3C9A"');
    expect(html).toContain("Serving");
    expect(html).toContain("singulared/Ornith-1.0-35B-MTP-GGUF:Q8_0");
    expect(html).toContain("v0.73.1");
    expect(html).toContain("128 GB");
    expect(html).toContain("99%");
    expect(html).toContain("32% mesh share");
    expect(html).toContain("--mesh-popover-x:82%");
    expect(html).toContain("mesh-node-popover above");
  });
});
