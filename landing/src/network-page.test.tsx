import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { NetworkPage, deriveDailyJobs, deriveNetworkMetrics, nodeState, type NetworkSnapshot, type NetworkTelemetryHistory } from "./NetworkPage";
import {
  BORDER_LINES,
  LAND_POINTS,
  buildArc,
  placeWorker,
  rotatePoint,
  sphere,
  type Vec3,
} from "./network-globe";
import { resolveLandingSurface } from "./routing";

const snapshot: NetworkSnapshot = {
  capturedAt: new Date().toISOString(),
  version: "0.2.54",
  summary: { registered: 3, connected: 2, online: 2, mobile: 0, offeredVramMb: 57_344, completedJobs: 7 },
  workers: [
    {
      id: "worker-madrid-1",
      kind: "desktop",
      status: "online",
      connected: true,
      region: "Madrid",
      offeredVramMb: 24_576,
      reliability: .98,
      jobsCompleted: 7,
      lastSeenAt: new Date().toISOString(),
      gpus: [{ id: "gpu-1", vendor: "NVIDIA", model: "RTX 4090", offeredVramMb: 24_576 }],
      deployments: [{ deploymentId: "dep-1", model: "real-model", mode: "replica", freeSlots: 1 }],
    },
    {
      id: "worker-standby-2",
      kind: "desktop",
      status: "online",
      connected: true,
      region: "Nuremberg",
      offeredVramMb: 32_768,
      reliability: .91,
      jobsCompleted: 3,
      lastSeenAt: new Date().toISOString(),
      gpus: [{ id: "gpu-2", vendor: "NVIDIA", model: "RTX 5090", offeredVramMb: 32_768 }],
      deployments: [],
    },
    {
      id: "worker-offline-3",
      kind: "desktop",
      status: "offline",
      connected: false,
      region: "auto",
      offeredVramMb: 0,
      reliability: .8,
      jobsCompleted: 2,
      lastSeenAt: new Date(Date.now() - 86_400_000).toISOString(),
      gpus: [],
      deployments: [],
    },
  ],
  models: [{ id: "real-model", replicas: 1, pipelines: 0 }],
  jobs: [{ id: "job-1", model: "real-model", status: "completed", workerId: "worker-madrid-1", inputTokens: 12, outputTokens: 25, updatedAt: new Date().toISOString() }],
};

/** Multi-day history with a counter reset, shared by the chart-wall tests. */
const wallHistory: NetworkTelemetryHistory = {
  capturedAt: "2026-08-14T00:20:00Z",
  intervalMinutes: 10,
  retentionDays: 90,
  range: "30d",
  samples: [
    { capturedAt: "2026-08-13T00:00:00Z", registeredNodes: 3, connectedNodes: 2, onlineNodes: 2, browserNodes: 0, activeModels: 1, modelReplicas: 1, modelPipelines: 0, offeredVramMb: 57_344, freeVramMb: 24_576, inflightJobs: 1, runningJobs: 1, completedJobs: 100 },
    { capturedAt: "2026-08-13T00:10:00Z", registeredNodes: 3, connectedNodes: 2, onlineNodes: 2, browserNodes: 0, activeModels: 1, modelReplicas: 1, modelPipelines: 0, offeredVramMb: 57_344, freeVramMb: 24_576, inflightJobs: 0, runningJobs: 0, completedJobs: 104 },
    { capturedAt: "2026-08-14T00:00:00Z", registeredNodes: 3, connectedNodes: 3, onlineNodes: 3, browserNodes: 0, activeModels: 1, modelReplicas: 1, modelPipelines: 0, offeredVramMb: 57_344, freeVramMb: 20_480, inflightJobs: 2, runningJobs: 2, completedJobs: 111 },
    // Counter reset: the negative delta must be ignored, not synthesised.
    { capturedAt: "2026-08-14T00:10:00Z", registeredNodes: 3, connectedNodes: 3, onlineNodes: 3, browserNodes: 0, activeModels: 1, modelReplicas: 1, modelPipelines: 0, offeredVramMb: 57_344, freeVramMb: 20_480, inflightJobs: 1, runningJobs: 1, completedJobs: 50 },
  ],
};

/** Rotates a lon/lat pair the way the renderer does, for assertion convenience. */
function project(lon: number, lat: number, yaw = 0, tilt = 0): Vec3 {
  const out: Vec3 = [0, 0, 0];
  return rotatePoint(sphere(lon, lat), 0, Math.sin(yaw), Math.cos(yaw), Math.sin(tilt), Math.cos(tilt), out);
}

describe("network observatory", () => {
  it("derives displayed capacity only from the public snapshot", () => {
    expect(deriveNetworkMetrics(snapshot)).toMatchObject({
      regions: 2,
      gpuCount: 2,
      memoryGb: 56,
      availableModels: 1,
      tokensServed: 37,
    });
  });

  it("classifies node state from real connection, status and deployments", () => {
    expect(nodeState(snapshot.workers[0]!)).toBe("serving");
    expect(nodeState(snapshot.workers[1]!)).toBe("standby");
    expect(nodeState(snapshot.workers[2]!)).toBe("joining");
  });

  it("ignores the placeholder 'auto' region when counting regions", () => {
    const autoOnly = { ...snapshot, workers: [{ ...snapshot.workers[0]!, region: "auto" }] };
    expect(deriveNetworkMetrics(autoOnly).regions).toBe(0);
  });

  it("places known regions at their real coordinates and flags unknown ones", () => {
    const madrid = placeWorker("Madrid", "worker-madrid-1");
    expect(madrid.located).toBe(true);
    expect(madrid.country).toBe("Spain");
    expect(madrid.lat).toBeCloseTo(40.4, 1);
    expect(madrid.lon).toBeCloseTo(-3.7, 1);

    const unknown = placeWorker("auto", "worker-offline-3");
    expect(unknown.located).toBe(false);
    expect(unknown.country).toBeNull();
    // Unlocated nodes must still be deterministic across renders.
    expect(placeWorker("auto", "worker-offline-3")).toEqual(unknown);
  });

  it("projects only the near hemisphere toward the viewer", () => {
    expect(project(0, 0)[2]).toBeCloseTo(1, 5);
    expect(project(180, 0)[2]).toBeCloseTo(-1, 5);
    // Yaw carries a meridian around to face the camera.
    expect(project(-90, 0, Math.PI / 2)[2]).toBeCloseTo(1, 5);
  });

  it("uses real land geometry for the globe surface", () => {
    const count = LAND_POINTS.length / 3;
    expect(count).toBeGreaterThan(10_000);

    const near = (lon: number, lat: number, tolerance: number) => {
      const [tx, ty, tz] = sphere(lon, lat);
      for (let index = 0; index < count; index += 1) {
        const dot = LAND_POINTS[index * 3]! * tx
          + LAND_POINTS[index * 3 + 1]! * ty
          + LAND_POINTS[index * 3 + 2]! * tz;
        if (dot > Math.cos(tolerance * (Math.PI / 180))) return true;
      }
      return false;
    };

    expect(near(-3.7, 40.4, 2)).toBe(true);    // Madrid
    expect(near(133, -24, 3)).toBe(true);      // central Australia
    expect(near(-40, 30, 2)).toBe(false);      // mid Atlantic
    expect(near(-140, -16, 3)).toBe(false);    // open South Pacific
  });

  it("keeps antimeridian landmasses from smearing a band across the ocean", () => {
    // Fiji's ring crosses +/-180, so a planar point-in-polygon test used to
    // report the whole -16deg latitude band as land and drew a dotted line
    // straight across the empty South Atlantic and Pacific.
    const count = LAND_POINTS.length / 3;
    const inBox = (lonMin: number, lonMax: number, latMin: number, latMax: number) => {
      let hits = 0;
      for (let index = 0; index < count; index += 1) {
        const x = LAND_POINTS[index * 3]!;
        const y = LAND_POINTS[index * 3 + 1]!;
        const z = LAND_POINTS[index * 3 + 2]!;
        const lon = Math.atan2(x, z) * (180 / Math.PI);
        const lat = Math.asin(Math.max(-1, Math.min(1, y))) * (180 / Math.PI);
        if (lon > lonMin && lon < lonMax && lat > latMin && lat < latMax) hits += 1;
      }
      return hits;
    };
    expect(inBox(-33, -12, -19, -14)).toBe(0);      // South Atlantic
    expect(inBox(-140, -100, -19, -14)).toBe(0);    // South Pacific
    expect(inBox(115, 150, -38, -15)).toBeGreaterThan(100);  // Australia survives
  });

  it("draws country borders as separate polylines", () => {
    expect(BORDER_LINES.length).toBeGreaterThan(100);
    for (const line of BORDER_LINES.slice(0, 50)) {
      expect(line.length % 3).toBe(0);
      expect(line.length / 3).toBeGreaterThanOrEqual(2);
      // Every border vertex must sit on the unit sphere.
      const magnitude = Math.hypot(line[0]!, line[1]!, line[2]!);
      expect(magnitude).toBeCloseTo(1, 4);
    }
  });

  it("lifts route arcs off the surface so they read as hops", () => {
    const arc = buildArc(sphere(-3.7, 40.4), sphere(11.1, 49.5), 28);
    expect(arc.length / 3).toBe(28);
    const magnitudeAt = (index: number) =>
      Math.hypot(arc[index * 3]!, arc[index * 3 + 1]!, arc[index * 3 + 2]!);
    expect(magnitudeAt(0)).toBeCloseTo(1, 3);
    expect(magnitudeAt(27)).toBeCloseTo(1, 3);
    expect(magnitudeAt(14)).toBeGreaterThan(1.1);
  });

  it("renders real metrics and an accessible globe without invented telemetry", () => {
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} animate={false} />);
    expect(html).toContain("GPUS ONLINE");
    expect(html).toContain("serving");
    expect(html).toContain("56.0");
    expect(html).toContain("Interactive globe showing 3 announced Mycellios nodes");
    // The reference page's simulated figures must never appear here.
    expect(html).not.toContain("1,248");
    expect(html).not.toContain("4,230,781");
    expect(html).not.toContain("tok/s");
  });

  it("renders persisted evidence beneath the live globe", () => {
    const history: NetworkTelemetryHistory = {
      capturedAt: new Date().toISOString(), intervalMinutes: 10, retentionDays: 90, range: "30d",
      samples: [{ capturedAt: new Date().toISOString(), registeredNodes: 3, connectedNodes: 2, onlineNodes: 2, browserNodes: 0, activeModels: 1, modelReplicas: 1, modelPipelines: 0, offeredVramMb: 57_344, freeVramMb: 24_576, inflightJobs: 1, runningJobs: 1, completedJobs: 7 }],
    };
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} initialHistory={history} animate={false} />);
    expect(html).toContain("A network you can inspect");
    expect(html).toContain("1 persisted observations");
    expect(html).toContain("Nodes behind the snapshot");
    expect(html).toContain("Deliberately absent");
  });

  it("derives daily jobs from positive deltas of the cumulative counter", () => {
    expect(deriveDailyJobs(wallHistory.samples)).toEqual([
      { day: "2026-08-13", count: 4 },
      { day: "2026-08-14", count: 7 },
    ]);
    // No samples means no fabricated baseline buckets.
    expect(deriveDailyJobs([])).toEqual([]);
  });

  it("renders the reference-grade chart wall from persisted samples only", () => {
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} initialHistory={wallHistory} animate={false} />);
    expect(html).toContain('id="net-wall-network-title"');
    expect(html).toContain("jobs completed per day");
    expect(html).toContain("derived from the persisted cumulative counter");
    expect(html).toContain("capacity offered (GB)");
    expect(html).toContain("nodes online");
    expect(html).toContain("workers online");
    expect(html).toContain("sampled every 10 min");
    // Striped bars render as stacks of thin rects inside a bronze group.
    expect(html).toContain('class="net-wall-bars"');
    const stripedBars = html.match(/<g class="net-wall-bars">([\s\S]*?)<\/g><text class="axis"/)?.[1] ?? "";
    expect(stripedBars).toContain("<rect");
    expect((stripedBars.match(/<rect/g) ?? []).length).toBeGreaterThan(2);
    // The step-area chart fills with a dot-grid pattern, the step line with none.
    expect(html).toContain("net-wall-dots");
    expect(html).toMatch(/<path class="net-wall-area bronze"[^>]*fill="url\(#/);
    expect(html).toMatch(/<path class="net-wall-line ivory"[^>]*><\/path>/);
    // Axis ticks and date labels come with every frame.
    expect(html).toContain("08-13");
    expect(html).toContain("08-14");
  });

  it("renders the snapshot stat grid from the live snapshot", () => {
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} initialHistory={wallHistory} animate={false} />);
    expect(html).toContain('id="net-wall-snapshot-title"');
    expect(html).toContain("net-wall-stat");
    expect(html).toContain("registered nodes");
    expect(html).toContain("completed jobs");
    expect(html).toContain("GPUs online");
  });

  it("keeps the $SPORE wall explicitly inactive without any economic number", () => {
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} initialHistory={wallHistory} animate={false} />);
    expect(html).toContain('id="net-wall-spore-title"');
    expect(html).toContain("Not active");
    expect(html).toContain("Not published");
    expect(html).toContain("economic feed not published yet");
  });

  it("lists the 8 most recent snapshot jobs as mono rows", () => {
    const manyJobs = {
      ...snapshot,
      jobs: Array.from({ length: 9 }, (_, index) => ({
        id: `job-${index}`,
        model: "real-model",
        status: "completed",
        workerId: "worker-madrid-1",
        inputTokens: index,
        outputTokens: 0,
        updatedAt: new Date(Date.now() - (9 - index) * 60_000).toISOString(),
      })),
    };
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={manyJobs} initialHistory={wallHistory} animate={false} />);
    expect(html).toContain('id="net-wall-activity-title"');
    expect(html).toContain("net-wall-activity-table");
    // job-0 is the oldest of nine and must be cut by the top-8 rule.
    expect(html).toContain("job-8");
    expect(html).not.toContain("job-0");
  });

  it("states honestly when the snapshot carries no jobs", () => {
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={{ ...snapshot, jobs: [] }} initialHistory={wallHistory} animate={false} />);
    expect(html).toContain("No jobs observed in the current snapshot");
  });

  it("styles the chart wall with dedicated forest-panel rules", () => {
    const css = readFileSync(new URL("./network-page.css", import.meta.url), "utf8");
    for (const selector of [".net-wall-grid", ".net-wall-panel", ".net-wall-stats", ".net-wall-stat", ".net-wall-bars", ".net-wall-dots", ".net-wall-tip", ".net-wall-activity-row"]) {
      expect(css).toContain(selector);
    }
    // The evidence base stays deep forest; pure black is a non-goal.
    expect(css).not.toMatch(/background:\s*#000/);
  });

  it("keeps the Overview console centered instead of stretching edge-to-edge", () => {
    const css = readFileSync(new URL("./network-page.css", import.meta.url), "utf8");
    const dashboardRule = css.match(/\.net-dashboard\s*\{([\s\S]*?)\}/)?.[1] ?? "";
    expect(dashboardRule).toContain("left: 50%");
    expect(dashboardRule).toContain("width: min(1200px");
    expect(dashboardRule).toContain("transform: translateX(-50%)");
  });

  it("uses three visual rows and keeps the globe to one grid cell", () => {
    const css = readFileSync(new URL("./network-page.css", import.meta.url), "utf8");
    const visualsRule = css.match(/\.net-dashboard-visuals\s*\{([\s\S]*?)\}/)?.[1] ?? "";
    const globeRule = css.match(/\.net-globe-card\s*\{([\s\S]*?)\}/)?.[1] ?? "";
    expect(visualsRule).toContain("grid-template-columns: repeat(2");
    expect(visualsRule).toContain("grid-template-rows: repeat(3");
    expect(globeRule).toContain("grid-row: auto");
  });

  it("offers one overview and one immersive globe mode without duplicating routes", () => {
    const overview = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} initialViewMode="overview" animate={false} />);
    const globe = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} initialViewMode="globe" animate={false} />);
    expect(overview).toContain("completed jobs");
    expect(overview).not.toContain("Waiting for verified data");
    expect(overview).toContain("Offered capacity");
    expect(overview).toContain("Node availability");
    expect(overview).toContain("Model fabric");
    expect(overview).toContain("Inference load");
    expect(overview).toContain("Completed work");
    expect(overview).toContain("LIVE TOPOLOGY");
    expect(overview).toContain("The network, in signals.");
    expect(overview).toContain("Not published");
    expect(overview).toContain("$SPORE");
    expect(globe).not.toContain("net-hero-overview");
    expect(globe).toContain("TOKENS IN SNAPSHOT");
  });

  it("uses the canonical landing header on the public observatory", () => {
    const html = renderToStaticMarkup(<NetworkPage initialSnapshot={snapshot} animate={false} />);
    expect(html).toContain("Live network");
    expect(html).toContain("$ SPORE");
    expect(html).toContain("mycellios on GitHub");
    expect(html).toContain("aria-current=\"page\"");
    expect(html).toContain("class=\"rb-brand\" href=\"/\"");
  });

  it("does not animate canvas geometry between overview and globe", () => {
    const css = readFileSync(new URL("./network-page.css", import.meta.url), "utf8");
    const globeRule = css.match(/\.net-globe\s*\{([^}]*)\}/)?.[1] ?? "";
    expect(globeRule).toContain("transition: opacity");
    expect(globeRule).not.toMatch(/transition:[^;]*(?:left|width|height|top)/);
  });

  it("renders a truthful empty state", () => {
    const html = renderToStaticMarkup(<NetworkPage animate={false} initialSnapshot={{
      ...snapshot,
      summary: { ...snapshot.summary, registered: 0, connected: 0, online: 0, offeredVramMb: 0 },
      workers: [], models: [], jobs: [],
    }} />);
    expect(html).toContain("No nodes announced");
    expect(html).toContain("Waiting for the first node");
  });

  it("routes bare Network separately while preserving every Panel view", () => {
    expect(resolveLandingSurface("/network", "")).toBe("network");
    expect(resolveLandingSurface("/network", "?view=inference")).toBe("panel");
    expect(resolveLandingSurface("/network", "?view=overview")).toBe("panel");
    expect(resolveLandingSurface("/admin", "")).toBe("panel");
    expect(resolveLandingSurface("/", "")).toBe("landing");
  });
});
