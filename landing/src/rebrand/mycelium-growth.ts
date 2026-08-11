import { GLOBE_LAND_POINTS } from "./globe-points";
import { makeRandom } from "./seeded-random";

/*
 * Growth simulation for the hero globe.
 *
 * Kept separate from the canvas renderer so the behaviour that actually
 * matters — the colony spreads over land, slowly, without ever stalling — can
 * be unit tested without a DOM.
 */

export const GROWTH_SECONDS = 6.5; // base time for one filament to finish extending
/*
 * Oldest hyphae retire past this so growth can continue forever.
 *
 * Raised from 132 once tips started holding a course. A fork needs three
 * filaments at one point — the stem plus its two children — so a cap that
 * retires the stem while the children are still growing converts every
 * junction back into a bend: measured at 132, a two-minute-old colony contained
 * *zero* points of degree three, having had fifteen at thirty seconds. The
 * network was a set of disjoint cords, which is the tidier cousin of the same
 * complaint. At 190 the frontier keeps its branch structure long enough to read
 * as a mat, and the draw cost is still a few hundred short strokes a frame.
 */
export const MAX_FILAMENTS = 190;
/* A fruiting body takes far longer to appear than a filament takes to extend:
   it is the payoff of the colony, and one that popped up in a couple of
   seconds would read as a marker being dropped, not as something growing. */
export const FRUIT_SECONDS = 26;
/*
 * Raised together with the heading bonus. Once tips hold a course they stop
 * doubling back over ground they have already colonised, so the same number of
 * them covers far more of the map and the colony thins out into a few long
 * runners. More tips, and more forking, puts the density back — a mycelium is a
 * branching mat, not a set of parallel cables.
 */
const MAX_TIPS = 16;
const BRANCH_CHANCE = 0.36;
/** Hops a tip makes before exhausting and freeing its slot for a branch. */
const MIN_TIP_LIFE = 6;
const MAX_TIP_LIFE = 16;
/*
 * Rolled per hub, so the colony's mushroom count is `hubs × this`.
 *
 * Tuned for what the *hero shows*, not for what the model holds, because a
 * sphere hides half of itself and the limb foreshortens most of the rest: of
 * everything standing on the globe only about a third reads at full size on
 * screen. At 0.28 on a one-in-seven hub rate the colony settled at seven
 * bodies and put **two** in front of the viewer — and a single mushroom among
 * four thousand land dots does not read as a motif, it reads as an artefact
 * someone forgot to remove. That was the whole complaint, and it was correct.
 *
 * At 0.55 on a one-in-four hub rate the colony carries around twenty-five and
 * shows a dozen: enough that fruiting is obviously a recurring event on this
 * network rather than a one-off, while still leaving the great majority of
 * hubs bare. The ceiling is set by the land dots, not by taste — neighbouring
 * dots sit ~11px apart and a mature body is ~23px, so past roughly forty
 * bodies the caps start overlapping each other and the continents silt up into
 * solid bronze. Rendering one body per land point was tried directly: the
 * landmasses fill in completely, no individual mushroom survives, and the
 * frame rate falls to 16fps. Density is what was wrong here, not rarity.
 */
const FRUIT_CHANCE = 0.55;
const MIN_HOP = 0.055; // radians ≈ 3.1° of arc
const MAX_HOP = 0.2; // radians ≈ 11.5° of arc

/*
 * A hypha does not travel in a straight line, and a network drawn out of
 * straight lines is a graph — the exact thing this hero is trying not to be.
 * Every filament therefore carries a bow: a lateral wander, expressed as a
 * fraction of its own hop length, applied inside the plane tangent to the
 * sphere. Kept as a fraction rather than as an absolute angle so a short hop
 * meanders less than a long one, which is what stops the short ones from
 * curling into loops.
 */
const MIN_BOW = 0.09;
const MAX_BOW = 0.24;

/*
 * Retirement used to be a splice: the oldest hop vanished between two frames.
 * On a globe that turns this slowly, something disappearing in one frame is the
 * single most visible event on screen — it reads as a glitch, not as a colony
 * ageing. Retiring hyphae now dim over this many seconds instead, which is long
 * enough that the eye never catches the moment one leaves.
 */
export const FADE_SECONDS = 4.5;

/*
 * Side hairs.
 *
 * The thing that separates mycelium from a node graph is that its filaments are
 * *hairy*: every length of hypha puts out short lateral branches that go
 * nowhere. They carry no state and connect to nothing — they exist so the
 * network has texture between its junctions rather than being clean strokes
 * between dots.
 */
const HAIR_CHANCE = 0.62;
const SECOND_HAIR_CHANCE = 0.34;
const HAIR_SECONDS = 3.2;

export interface Vector {
  x: number;
  y: number;
  z: number;
}

/**
 * A short lateral branch off a filament, growing nowhere.
 *
 * `at` is where along the parent it attaches (0…1), `side` which way it leaves,
 * `length` how far it reaches as a fraction of the parent's hop. Real hyphae
 * are covered in these; without them a filament is a stroke between two dots.
 */
export interface Hair {
  at: number;
  side: number;
  length: number;
  /** 0…1 — hairs sprout after the parent has passed their attachment point. */
  progress: number;
  speed: number;
}

export interface Filament {
  from: Vector;
  to: Vector;
  progress: number;
  speed: number;
  hub: boolean;
  /**
   * Lateral bow, as a signed fraction of the hop's own length. Zero would be a
   * great-circle arc — a straight line on a sphere — which is what made the
   * network read as a wire diagram.
   */
  bow: number;
  /** Wander added on top of the bow, so no two filaments share a curve. */
  wobble: number;
  hairs: Hair[];
  /**
   * 1 while the filament is part of the live colony, falling to 0 as it
   * retires. Deleted only once it reaches 0, so nothing ever pops off screen.
   */
  vitality: number;
  /** Set once the hub has been offered its single chance to fruit. */
  settled?: boolean;
}

/*
 * A mushroom on the surface of the globe.
 *
 * Hubs are the filaments the renderer already draws in bronze — the ones the
 * colony has committed to. A few of them go on to fruit, which is the visual
 * claim the whole hero makes: the network is not just spreading, it is
 * producing. `maturity` runs 0→1 over FRUIT_SECONDS and then stops.
 */
export interface Fruit {
  /** Land point it grew from; also the surface normal, since land is a unit vector. */
  at: Vector;
  /** The hop that produced it. When that retires, so does this. */
  source: Filament;
  maturity: number;
  speed: number;
  /**
   * -1…1 tilt off the normal, so no two stand to attention.
   *
   * There is deliberately no companion "which way is it turned" field: a
   * mushroom is a solid of revolution about its own stalk, so its outline is
   * the same from every side, and the renderer orients the body across the
   * line of sight rather than at a stored angle.
   */
  lean: number;
  /** Radial lines under the cap. Small variation keeps them from cloning. */
  gills: number;
}

interface Tip {
  at: Vector;
  wait: number;
  /** Direction of travel, carried between hops so the tip keeps heading. */
  heading: Vector | null;
  /**
   * Hops this tip has left before it stops extending.
   *
   * Without this the tip pool fills to MAX_TIPS within about thirty seconds and
   * then never changes: every slot is occupied by an immortal tip, the branch
   * roll can never find room, and the colony spends the rest of the session
   * extending the same handful of chains end to end. Measured, that took a
   * network with fifteen forks at thirty seconds down to zero by two minutes.
   * Real hyphal tips exhaust and hand growth on to their branches, and that
   * turnover is what keeps a mat branching instead of merely getting longer.
   */
  life: number;
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

function normalize(v: Vector): Vector {
  const length = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / length, y: v.y / length, z: v.z / length };
}

function cross(a: Vector, b: Vector): Vector {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

/**
 * The direction that runs sideways across a hop, tangent to the sphere.
 *
 * This is `from × to` — the axis the great circle turns about — and nothing
 * else, which is worth stating because the obvious-looking alternative is
 * silently wrong. At a point P on the arc the three orthogonal directions are P
 * itself (straight up out of the surface), the forward tangent `axis × P`, and
 * the sideways one. Crossing the axis back with P gives the *forward* tangent,
 * so displacing along it slides the point further along the very same great
 * circle: the geometry still passes a "moved away from slerp" check while
 * drawing an identical straight arc. That shipped, and the filaments stayed
 * visibly straight until the curve was rasterised and looked at.
 *
 * The axis is already perpendicular to P for any point on the arc, so it is
 * tangent to the sphere there, and moving along it is the only displacement
 * that actually bends the path.
 */
function sideways(from: Vector, to: Vector): Vector {
  const axis = cross(from, to);
  const length = Math.hypot(axis.x, axis.y, axis.z);
  // Degenerate hop (endpoints coincident or antipodal): no meaningful side.
  if (length < 1e-9) return { x: 0, y: 0, z: 0 };
  return { x: axis.x / length, y: axis.y / length, z: axis.z / length };
}

/** Unit forward direction along a hop at parameter `t`. */
function forwardAt(from: Vector, to: Vector, at: Vector): Vector {
  const axis = cross(from, to);
  const length = Math.hypot(axis.x, axis.y, axis.z);
  if (length < 1e-9) return { x: 0, y: 0, z: 0 };
  return normalize(cross({ x: axis.x / length, y: axis.y / length, z: axis.z / length }, at));
}

/**
 * The direction a filament is travelling as it arrives at its far end.
 *
 * Growth uses this to keep heading: a tip that picks its next hop at random
 * turns a corner at every junction, and a chain of corners is a zigzag, not a
 * hypha. Taken from the bowed curve rather than from the chord so the heading
 * matches the line that was actually drawn.
 */
export function filamentHeading(filament: Filament): Vector {
  const end = filamentPoint(filament, 1);
  const before = filamentPoint(filament, 0.94);
  const raw = { x: end.x - before.x, y: end.y - before.y, z: end.z - before.z };
  if (Math.hypot(raw.x, raw.y, raw.z) < 1e-9) return forwardAt(filament.from, filament.to, filament.to);
  // Project onto the tangent plane at the endpoint, so the heading is a
  // direction along the surface rather than a chord through the sphere.
  const radial = dot(raw, end);
  const tangent = { x: raw.x - end.x * radial, y: raw.y - end.y * radial, z: raw.z - end.z * radial };
  if (Math.hypot(tangent.x, tangent.y, tangent.z) < 1e-9) return { x: 0, y: 0, z: 0 };
  return normalize(tangent);
}

/**
 * A point along a filament, at `t` of the way from its start to its end.
 *
 * This is the whole difference between a mycelium and a mesh, so it is one
 * function shared by the renderer and the tests rather than a curve the canvas
 * improvises. The great-circle position is displaced sideways by a lobe that is
 * zero at both ends — so junctions still meet exactly — and widest in the
 * middle, plus a second, faster wobble that keeps the curve from being a clean
 * parabola. Both scale with the hop's own angular length, so a short hop is a
 * gentle kink and a long one visibly meanders.
 */
export function filamentPoint(filament: Filament, t: number): Vector {
  const base = slerp(filament.from, filament.to, t);
  if (filament.bow === 0 && filament.wobble === 0) return base;
  const side = sideways(filament.from, filament.to);
  const span = Math.acos(Math.min(1, Math.max(-1, dot(filament.from, filament.to))));
  // sin(πt) is zero at both ends and 1 at the middle: a bow that cannot move
  // either endpoint, however hard it is driven.
  const lobe = Math.sin(Math.PI * t);
  const offset = span * (filament.bow * lobe + filament.wobble * Math.sin(Math.PI * 2.7 * t) * lobe);
  return normalize({
    x: base.x + side.x * offset,
    y: base.y + side.y * offset,
    z: base.z + side.z * offset,
  });
}

/**
 * Where a hair reaches, given its parent.
 *
 * Hairs leave the parent sideways and slightly forward — a branch that leaves
 * at a right angle reads as a tick mark on a ruler. The forward component comes
 * from the parent's own direction at the attachment point, so the hair sweeps
 * back along the filament the way a real lateral branch does.
 */
export function hairPoint(filament: Filament, hair: Hair, t: number): Vector {
  const root = filamentPoint(filament, hair.at);
  const ahead = filamentPoint(filament, Math.min(1, hair.at + 0.06));
  const forward = normalize({ x: ahead.x - root.x, y: ahead.y - root.y, z: ahead.z - root.z });
  const side = sideways(filament.from, filament.to);
  const span = Math.acos(Math.min(1, Math.max(-1, dot(filament.from, filament.to))));
  const reach = span * hair.length * t;
  // Mostly sideways, a third forward: a lateral that leans downstream.
  return normalize({
    x: root.x + (side.x * hair.side * 0.94 + forward.x * 0.34) * reach,
    y: root.y + (side.y * hair.side * 0.94 + forward.y * 0.34) * reach,
    z: root.z + (side.z * hair.side * 0.94 + forward.z * 0.34) * reach,
  });
}

export class Mycelium {
  readonly filaments: Filament[] = [];
  /** Mushrooms currently on the globe. Bounded by the hub count, so no cap. */
  readonly fruits: Fruit[] = [];
  private readonly tips: Tip[] = [];
  private readonly random: () => number;

  constructor(private readonly land: Vector[], seed = 0x9e3779b9) {
    this.random = makeRandom(seed);
    // Seeds sit on separate landmasses so growth starts from several colonies.
    for (const [lat, lng] of [[40, -3], [37, 127], [-23, -46], [52, 13]] as const) {
      this.tips.push({
        at: this.nearestLand(toVector(lat, lng)),
        wait: this.random() * 2,
        heading: null,
        life: this.tipLife(),
      });
    }
  }

  private tipLife(): number {
    return MIN_TIP_LIFE + Math.floor(this.random() * (MAX_TIP_LIFE - MIN_TIP_LIFE));
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

  /**
   * Picks a land point a short hop away, preferring the closer candidates and —
   * crucially — the ones that continue the direction the tip is already going.
   *
   * Without the heading term this scored on proximity alone, so each hop left
   * at an angle unrelated to the one before it. Individually every filament was
   * a fine short arc; strung together they drew a zigzag of hard corners, and a
   * chain of corners reads as a routing diagram no matter how the individual
   * segments are curved. A hypha, by contrast, has momentum: it keeps going and
   * turns gradually. That is what the heading bonus buys, and it is worth more
   * to this than the per-filament bow is.
   */
  private nextHop(from: Vector, heading: Vector | null): Vector | null {
    let best: Vector | null = null;
    let bestScore = -Infinity;
    const hasHeading = !!heading && Math.hypot(heading.x, heading.y, heading.z) > 1e-9;
    /*
     * Scan the whole land grid rather than sampling it randomly.
     *
     * The old loop drew 46 random points out of 4,607 and kept those inside the
     * hop band. Only about 105 of the grid is in that band from any given
     * point, so those 46 draws yielded — measured — an average of *one*
     * qualifying candidate. Every scoring term downstream was therefore
     * decorative: there was nothing to choose between, and the hop went
     * wherever the single survivor happened to be, which is precisely the
     * random-walk zigzag the heading bonus exists to remove. The direction
     * preference only becomes real once there is a field of candidates to
     * prefer within.
     *
     * A full scan is ~4.6k dot products, and it runs once per hop — a few times
     * a second across the whole colony, not once per frame per filament.
     */
    for (const candidate of this.land) {
      const angle = Math.acos(Math.min(1, Math.max(-1, dot(from, candidate))));
      if (angle < MIN_HOP || angle > MAX_HOP) continue;
      let score = -angle + this.random() * 0.05;
      if (hasHeading) {
        // Direction from `from` towards the candidate, in the tangent plane.
        const raw = { x: candidate.x - from.x, y: candidate.y - from.y, z: candidate.z - from.z };
        const radial = dot(raw, from);
        const tangent = {
          x: raw.x - from.x * radial,
          y: raw.y - from.y * radial,
          z: raw.z - from.z * radial,
        };
        const length = Math.hypot(tangent.x, tangent.y, tangent.z);
        if (length > 1e-9) {
          // cos of the turn angle: 1 straight on, -1 doubling back.
          const alignment = dot(heading!, {
            x: tangent.x / length,
            y: tangent.y / length,
            z: tangent.z / length,
          });
          /*
           * Weighted above the proximity term it competes with — distance
           * scores span MAX_HOP − MIN_HOP ≈ 0.15 — but deliberately not so far
           * above it that direction becomes the only thing that matters. At
           * 0.28 the mean turn came out at cos 0.98, which is a straight line:
           * the colony stopped zigzagging and started drawing taut cables
           * across the continents instead, which is the same wrongness with the
           * opposite sign. At 0.15 the runners still hold a course over many
           * hops but visibly wander while doing it, which is what a hypha does.
           */
          score += alignment * 0.15;
        }
      }
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }
    return best;
  }

  /** The lateral branches a filament will put out as it extends. */
  private growHairs(): Hair[] {
    const hairs: Hair[] = [];
    const add = () => {
      hairs.push({
        // Never at the very ends: a hair on a junction reads as a bad join.
        at: 0.22 + this.random() * 0.56,
        side: this.random() < 0.5 ? -1 : 1,
        length: 0.22 + this.random() * 0.3,
        progress: 0,
        speed: 1 / (HAIR_SECONDS * (0.7 + this.random() * 0.9)),
      });
    };
    if (this.random() < HAIR_CHANCE) add();
    if (this.random() < SECOND_HAIR_CHANCE) add();
    return hairs;
  }

  private sprout(from: Vector, heading: Vector | null): boolean {
    const target = this.nextHop(from, heading);
    if (!target) return false;
    const bowSize = MIN_BOW + this.random() * (MAX_BOW - MIN_BOW);
    this.filaments.push({
      from,
      to: target,
      progress: 0,
      speed: 1 / (GROWTH_SECONDS * (0.7 + this.random() * 0.8)),
      // Hubs are where the network commits, and they are the only places that
      // can fruit — so this rate is the real ceiling on how many mushrooms the
      // globe can ever carry, not FRUIT_CHANCE.
      hub: this.random() < 0.26,
      bow: (this.random() < 0.5 ? -1 : 1) * bowSize,
      wobble: (this.random() - 0.5) * 0.09,
      hairs: this.growHairs(),
      vitality: 1,
    });
    return true;
  }

  /** Advances growth by `delta` seconds. */
  advance(delta: number): void {
    for (const filament of this.filaments) {
      // Hairs trail the tip: one only starts once the parent has grown past
      // where it attaches, so the texture appears in the wake of the advance
      // rather than all at once when the hop lands.
      for (const hair of filament.hairs) {
        if (hair.progress >= 1 || filament.progress < hair.at) continue;
        hair.progress = Math.min(1, hair.progress + hair.speed * delta);
      }

      if (filament.progress < 1) {
        filament.progress = Math.min(1, filament.progress + filament.speed * delta);
        continue;
      }
      /*
       * A filament that has just finished is a completed hop, and a hub that
       * completes is the colony committing to that point — so that is the one
       * moment it can fruit. `settled` makes the roll happen exactly once per
       * hub rather than on every frame after it lands, which would make
       * fruiting a function of frame rate.
       */
      if (filament.settled) continue;
      filament.settled = true;
      if (!filament.hub || this.random() >= FRUIT_CHANCE) continue;
      this.fruits.push({
        at: filament.to,
        source: filament,
        maturity: 0,
        speed: 1 / (FRUIT_SECONDS * (0.7 + this.random() * 0.8)),
        lean: (this.random() - 0.5) * 1.5,
        gills: 5 + Math.floor(this.random() * 4),
      });
    }

    for (const fruit of this.fruits) {
      if (fruit.maturity < 1) fruit.maturity = Math.min(1, fruit.maturity + fruit.speed * delta);
    }

    for (let i = this.tips.length - 1; i >= 0; i -= 1) {
      const tip = this.tips[i]!;
      tip.wait -= delta;
      if (tip.wait > 0) continue;

      if (!this.sprout(tip.at, tip.heading)) {
        // Exhausted corner of a landmass: restart this tip elsewhere, with no
        // inherited direction since it is no longer continuing anything.
        tip.at = this.land[Math.floor(this.random() * this.land.length)]!;
        tip.wait = 1 + this.random();
        tip.heading = null;
        tip.life = this.tipLife();
        continue;
      }

      const grown = this.filaments[this.filaments.length - 1]!;
      tip.at = grown.to;
      tip.heading = filamentHeading(grown);
      tip.wait = GROWTH_SECONDS * (0.55 + this.random() * 0.7);
      tip.life -= 1;

      if (this.tips.length < MAX_TIPS && this.random() < BRANCH_CHANCE) {
        /*
         * A branch leaves at an angle rather than continuing straight — that is
         * what makes it a branch. Inheriting the parent's heading unchanged
         * would grow two filaments along the same line and read as one thick
         * one; `null` lets the new tip choose freely on its first hop and pick
         * up its own momentum from there.
         */
        this.tips.push({
          at: grown.to,
          wait: this.random() * GROWTH_SECONDS,
          heading: null,
          life: this.tipLife(),
        });
      }

      /*
       * An exhausted tip gives up its slot. The colony must never run out of
       * frontier, though, so the last few tips are reseeded onto live territory
       * rather than removed — growth continues somewhere it has already reached
       * instead of the network freezing or restarting from nothing.
       */
      if (tip.life <= 0) {
        if (this.tips.length > 3) {
          this.tips.splice(i, 1);
        } else {
          const anchor = this.filaments[Math.floor(this.random() * this.filaments.length)];
          tip.at = anchor ? anchor.to : this.land[Math.floor(this.random() * this.land.length)]!;
          tip.heading = null;
          tip.wait = GROWTH_SECONDS * (0.4 + this.random() * 0.6);
          tip.life = this.tipLife();
        }
      }
    }

    /*
     * Retiring the oldest hyphae is what lets growth continue forever without
     * the geometry growing forever — but retirement is now a death, not a
     * deletion.
     *
     * The previous version spliced the oldest hop out of the array between two
     * frames. On a globe that turns at 0.025 rad/s, an element vanishing in
     * 16ms is by far the fastest thing on screen: the eye is caught by exactly
     * the event the composition least wants to advertise, and the network reads
     * as flickering rather than as growing. Now the oldest hyphae past the cap
     * are marked dying, dim over FADE_SECONDS, and are only dropped once they
     * are fully transparent — so the colony's edge recedes as quietly as it
     * advances.
     */
    let live = 0;
    for (const filament of this.filaments) if (filament.vitality >= 1) live += 1;
    for (const filament of this.filaments) {
      if (live <= MAX_FILAMENTS) break;
      if (filament.vitality < 1) continue;
      // Oldest first: `filaments` is append-ordered, so this walks from the
      // colony's origin outward.
      filament.vitality = 1 - 1e-6;
      live -= 1;
    }

    for (const filament of this.filaments) {
      if (filament.vitality >= 1) continue;
      filament.vitality = Math.max(0, filament.vitality - delta / FADE_SECONDS);
    }

    // A mushroom belongs to the hop that grew it, so it goes when that hop
    // finally leaves — otherwise a long session accumulates mushrooms standing
    // on filaments that are no longer drawn.
    for (let i = this.filaments.length - 1; i >= 0; i -= 1) {
      const filament = this.filaments[i]!;
      if (filament.vitality > 0) continue;
      this.filaments.splice(i, 1);
      if (!filament.hub) continue;
      // Matched by the filament itself, not by its endpoint: two hops can land
      // on the same land dot, and dropping by position would retire the wrong
      // mushroom — or one that is still standing on a live filament.
      const index = this.fruits.findIndex((fruit) => fruit.source === filament);
      if (index >= 0) this.fruits.splice(index, 1);
    }
  }
}
