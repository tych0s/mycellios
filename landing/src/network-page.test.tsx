import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { NetworkPage, deriveNetworkMetrics, nodeState, type NetworkSnapshot } from "./NetworkPage";
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
