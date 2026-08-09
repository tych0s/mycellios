#!/usr/bin/env node
/*
 * Generates the dot-globe land grid used by the rebrand hero.
 *
 * Input : a TopoJSON "land" topology (world-atlas land-110m.json).
 * Output: landing/src/rebrand/globe-points.ts, a compact half-degree grid of
 *         land coordinates encoded as integer pairs.
 *
 * Usage: node scripts/generate-globe-points.mjs <land-110m.json>
 *
 * The generated file is committed, so this script only needs to run when the
 * grid density changes. It has no runtime dependencies.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const STEP = 1.6; // degrees between latitude rings
const OUTPUT = resolve(process.cwd(), "landing/src/rebrand/globe-points.ts");

function decodeArcs(topology) {
  const { scale, translate } = topology.transform;
  return topology.arcs.map((arc) => {
    let x = 0;
    let y = 0;
    return arc.map(([dx, dy]) => {
      x += dx;
      y += dy;
      return [x * scale[0] + translate[0], y * scale[1] + translate[1]];
    });
  });
}

function ringFromArcIndexes(indexes, arcs) {
  const ring = [];
  for (const index of indexes) {
    const arc = index < 0 ? arcs[~index].slice().reverse() : arcs[index];
    for (let i = index < 0 || ring.length === 0 ? 0 : 1; i < arc.length; i += 1) ring.push(arc[i]);
  }
  return ring;
}

function polygonsFromTopology(topology) {
  const arcs = decodeArcs(topology);
  const geometry = topology.objects.land;
  const collection = geometry.type === "GeometryCollection" ? geometry.geometries : [geometry];
  const polygons = [];
  for (const item of collection) {
    const list = item.type === "MultiPolygon" ? item.arcs : [item.arcs];
    for (const polygon of list) polygons.push(polygon.map((ring) => ringFromArcIndexes(ring, arcs)));
  }
  return polygons;
}

function boundsOf(ring) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of ring) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return [minX, minY, maxX, maxY];
}

function pointInRing(lng, lat, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > lat !== yj > lat && lng < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

const topology = JSON.parse(readFileSync(process.argv[2] ?? "/tmp/land-110m.json", "utf8"));
const polygons = polygonsFromTopology(topology).map((rings) => ({
  rings,
  bounds: boundsOf(rings[0]),
}));

function isLand(lng, lat) {
  for (const { rings, bounds } of polygons) {
    if (lng < bounds[0] || lng > bounds[2] || lat < bounds[1] || lat > bounds[3]) continue;
    if (!pointInRing(lng, lat, rings[0])) continue;
    let hole = false;
    for (let i = 1; i < rings.length; i += 1) {
      if (pointInRing(lng, lat, rings[i])) {
        hole = true;
        break;
      }
    }
    if (!hole) return true;
  }
  return false;
}

// Rings of constant latitude with longitude spacing corrected by cos(lat) so
// the dots stay evenly spaced on the sphere instead of bunching at the poles.
const points = [];
for (let lat = -84; lat <= 84; lat += STEP) {
  const circumference = Math.cos((lat * Math.PI) / 180);
  if (circumference <= 0.02) continue;
  const count = Math.max(6, Math.round((360 / STEP) * circumference));
  for (let i = 0; i < count; i += 1) {
    const lng = -180 + (360 / count) * i;
    if (!isLand(lng, lat)) continue;
    points.push(Math.round(lat * 2), Math.round(lng * 2));
  }
}

const file = `// GENERATED FILE — do not edit by hand.
// Source: world-atlas land-110m TopoJSON, rasterized by scripts/generate-globe-points.mjs
// Format: flat pairs of half-degree integers [lat*2, lng*2, ...] covering land only.
// Dots are spaced by cos(latitude) so the globe grid stays visually even.

export const GLOBE_STEP_DEGREES = ${STEP};

export const GLOBE_LAND_POINTS: readonly number[] = [
${points.join(",").replace(/((?:-?\d+,){40})/g, "$1\n  ").replace(/^/gm, "  ").trimEnd()}
];
`;

writeFileSync(OUTPUT, file);
process.stdout.write(`${points.length / 2} land points -> ${OUTPUT}\n`);
