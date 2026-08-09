import { GLOBE_LAND_POINTS } from "./globe-points";

/*
 * Growth simulation for the hero globe.
 *
 * Kept separate from the canvas renderer so the behaviour that actually
 * matters — the colony spreads over land, slowly, without ever stalling — can
 * be unit tested without a DOM.
 */

export const GROWTH_SECONDS = 6.5; // base time for one filament to finish extending
export const MAX_FILAMENTS = 132; // oldest hyphae retire past this so growth continues
const MAX_TIPS = 11;
const BRANCH_CHANCE = 0.26;
const MIN_HOP = 0.055; // radians ≈ 3.1° of arc
const MAX_HOP = 0.2; // radians ≈ 11.5° of arc

export interface Vector {
  x: number;
  y: number;
  z: number;
}

export interface Filament {
  from: Vector;
  to: Vector;
  progress: number;
  speed: number;
  hub: boolean;
}

interface Tip {
  at: Vector;
  wait: number;
}

/** Deterministic RNG so the mycelium grows the same way on every load. */
function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function toVector(latDegrees: number, lngDegrees: number): Vector {
  const lat = latDegrees * (Math.PI / 180);
  const lng = lngDegrees * (Math.PI / 180);
  const cosLat = Math.cos(lat);
  return { x: cosLat * Math.sin(lng), y: Math.sin(lat), z: cosLat * Math.cos(lng) };
}

export function buildLandVectors(): Vector[] {
  const vectors: Vector[] = [];
  for (let i = 0; i < GLOBE_LAND_POINTS.length; i += 2) {
    vectors.push(toVector(GLOBE_LAND_POINTS[i]! / 2, GLOBE_LAND_POINTS[i + 1]! / 2));
  }
  return vectors;
}

export function dot(a: Vector, b: Vector): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** Great-circle interpolation, so filaments hug the surface of the sphere. */
export function slerp(a: Vector, b: Vector, t: number): Vector {
  const cosine = Math.min(1, Math.max(-1, dot(a, b)));
  const angle = Math.acos(cosine);
  if (angle < 1e-6) return a;
  const sine = Math.sin(angle);
  const wa = Math.sin((1 - t) * angle) / sine;
  const wb = Math.sin(t * angle) / sine;
  return { x: a.x * wa + b.x * wb, y: a.y * wa + b.y * wb, z: a.z * wa + b.z * wb };
}

export class Mycelium {
  readonly filaments: Filament[] = [];
  private readonly tips: Tip[] = [];
  private readonly random: () => number;

  constructor(private readonly land: Vector[], seed = 0x9e3779b9) {
    this.random = createRandom(seed);
    // Seeds sit on separate landmasses so growth starts from several colonies.
    for (const [lat, lng] of [[40, -3], [37, 127], [-23, -46], [52, 13]] as const) {
      this.tips.push({ at: this.nearestLand(toVector(lat, lng)), wait: this.random() * 2 });
    }
  }

  private nearestLand(target: Vector): Vector {
    let best = this.land[0]!;
    let bestDot = -2;
    for (const candidate of this.land) {
      const value = dot(candidate, target);
      if (value > bestDot) {
        bestDot = value;
        best = candidate;
      }
    }
    return best;
  }

  /** Picks a land point a short hop away, preferring the closer candidates. */
  private nextHop(from: Vector): Vector | null {
    let best: Vector | null = null;
    let bestScore = -Infinity;
    for (let attempt = 0; attempt < 46; attempt += 1) {
      const candidate = this.land[Math.floor(this.random() * this.land.length)]!;
      const angle = Math.acos(Math.min(1, Math.max(-1, dot(from, candidate))));
      if (angle < MIN_HOP || angle > MAX_HOP) continue;
      const score = -angle + this.random() * 0.05;
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }
    return best;
  }

  private sprout(from: Vector): boolean {
    const target = this.nextHop(from);
    if (!target) return false;
    this.filaments.push({
      from,
      to: target,
      progress: 0,
      speed: 1 / (GROWTH_SECONDS * (0.7 + this.random() * 0.8)),
      hub: this.random() < 0.16,
    });
    return true;
  }

  /** Advances growth by `delta` seconds. */
  advance(delta: number): void {
    for (const filament of this.filaments) {
      if (filament.progress < 1) filament.progress = Math.min(1, filament.progress + filament.speed * delta);
    }

    for (let i = this.tips.length - 1; i >= 0; i -= 1) {
      const tip = this.tips[i]!;
      tip.wait -= delta;
      if (tip.wait > 0) continue;

      if (!this.sprout(tip.at)) {
        // Exhausted corner of a landmass: restart this tip elsewhere.
        tip.at = this.land[Math.floor(this.random() * this.land.length)]!;
        tip.wait = 1 + this.random();
        continue;
      }

      const grown = this.filaments[this.filaments.length - 1]!;
      tip.at = grown.to;
      tip.wait = GROWTH_SECONDS * (0.55 + this.random() * 0.7);

      if (this.tips.length < MAX_TIPS && this.random() < BRANCH_CHANCE) {
        this.tips.push({ at: grown.to, wait: this.random() * GROWTH_SECONDS });
      }
    }

    while (this.filaments.length > MAX_FILAMENTS) this.filaments.shift();
  }
}
