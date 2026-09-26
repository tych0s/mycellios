import { useEffect, useRef } from "react";
import { growFruitingBody, type Point } from "./fruiting-body";
import {
  buildLandVectors,
  filamentPoint,
  type Fruit,
  GROWTH_SECONDS,
  hairPoint,
  Mycelium,
  type Vector,
} from "./mycelium-growth";

/*
 * Dot globe for the hero: the world drawn as land points, connected by a
 * mycelium that grows across continents instead of a fixed decorative mesh.
 *
 * Two motions run at once and both are deliberately slow:
 *  - the globe turns steadily eastward, so land always travels to the right;
 *  - filaments extend hypha by hypha from a few seeds, branching into nearby
 *    land, so the network is visibly larger a minute after the page loads.
 * Fruiting remains reserved for the single foreground specimen; repeating it
 * across the globe competes with that focal silhouette.
 *
 * The turn never reverses, but its speed does: parking the empty Pacific in
 * front of the viewer for a minute reads as a broken graphic, so the globe
 * accelerates through that hemisphere and slows again over the continents.
 *
 * Everything renders into a single canvas because a 4.6k-point globe plus the
 * growing mesh is far cheaper as one draw loop than as thousands of SVG nodes.
 */

const DRIFT_SPEED = 0.025; // radians/s while the continents face the viewer
const PACIFIC_BOOST = 2.4; // extra speed at the emptiest point of the ocean
const DRIFT_START = 0.35; // opens on Europe/Africa, then turns east from there
const PACIFIC_ROTATION = 2.95; // rotation at which the open ocean faces the viewer
// Positive tilt leans the north pole toward the viewer, so the camera looks
// slightly down onto the globe. A negative tilt hides the north pole behind the
// horizon and leaves only southern latitudes on screen.
// At 0.34 the visible centre sits at ~20°N and the globe reads as southern-only;
// at 0.52 the camera looks down on it hard enough to flatten the north. This is
// the middle: almost head-on, with just enough elevation to favour the north.
const TILT = 0.42; // radians — visible centre ~24°N
const PRE_GROWN_STEPS = 34; // the network is already alive on first paint
/* A bowed filament needs enough samples that its curve is a curve rather than
   a polyline. At 18 the chord error on the widest bow was visible as facets on
   a desktop hero; 26 puts it under half a pixel. */
const ARC_SEGMENTS = 26;
const HAIR_SEGMENTS = 6;
const TAU = Math.PI * 2;
const SHOW_GLOBE_FRUITING_BODIES = false;
/* A mature mushroom stands this fraction of the globe's radius off the
   surface. Tuned by measurement rather than by taste: at 0.028 a body standing
   away from the centre of the disc foreshortens to five or six screen pixels,
   which is a bronze speck — the silhouette that carries the whole idea is
   simply not resolvable. At 0.055 the same body lands around 14–20px on a
   desktop hero, which is where the cap and stipe become legible while the
   thing is still visibly smaller than a continent.

   Trimmed from 0.055 once the colony started carrying a dozen bodies on screen
   instead of two. Size and count trade against each other on a fixed sphere:
   land dots are ~11px apart, so at 0.055 (~23px) a dozen caps begin to touch
   and the continents read as bronze rather than as land with things standing
   on it. At 0.046 a mature body is ~19px — still comfortably above the ~14px
   where the cap and stipe stop resolving — and the fan of them reads as a
   population. */
const FRUIT_SCALE = 0.046;

/** Angular velocity: cruises over land, hurries across the empty Pacific. */
function driftSpeed(rotation: number): number {
  const bump = (1 + Math.cos(rotation - PACIFIC_ROTATION)) / 2;
  return DRIFT_SPEED * (1 + PACIFIC_BOOST * bump ** 3);
}

export function HeroGlobe() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const land = buildLandVectors();
    const mycelium = new Mycelium(land);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    // Start from an established colony instead of an empty planet.
    for (let i = 0; i < PRE_GROWN_STEPS; i += 1) mycelium.advance(GROWTH_SECONDS * 0.9);

    let width = 0;
    let height = 0;
    const resize = () => {
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      const rect = canvas.getBoundingClientRect();
      width = Math.max(1, rect.width);
      height = Math.max(1, rect.height);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    resize();

    const observer = new ResizeObserver(() => { resize(); render(); });
    observer.observe(canvas);

    let rotation = DRIFT_START;
    const cosTilt = Math.cos(TILT);
    const sinTilt = Math.sin(TILT);

    /*
     * A mushroom standing on the globe.
     *
     * The body is generated in its own flat frame (see `fruiting-body.ts`) and
     * mapped here onto the sphere: local +y becomes the surface normal at the
     * land point, local +x the tangent that runs across the line of sight.
     * Because both axes are real 3-D directions, the body stands up off the
     * surface, leans away from the pole and foreshortens toward the limb on its
     * own — but its width always faces the camera, so what is drawn is the
     * outline of a solid of revolution rather than a flat cut-out of one.
     *
     * The cap is filled rather than stroked. Drawn as an outline it is an open
     * arc that stops at each rim, and at the 12–20px these bodies actually
     * occupy that arc separates from the stalk and reads as a stray bronze
     * squiggle. Filling closes rim to rim, so the silhouette — a dome on a
     * stalk — survives at any size the globe puts it at.
     */
    const drawFruit = (
      fruit: Fruit,
      project: (vector: Vector, scale: number) => { sx: number; sy: number; depth: number },
      radius: number,
    ): void => {
      const base = project(fruit.at, 1);
      // Hidden behind the globe, or so close to the limb that the body would
      // be drawn edge-on through the horizon.
      if (base.depth <= 0.16) return;

      const normal = fruit.at;
      const size = FRUIT_SCALE;
      /*
       * The body is billboarded: built in screen space off its own footprint,
       * not extruded through the sphere's 3-D frame.
       *
       * Mapping local +y onto the world-space normal is the obvious thing and
       * it is wrong twice. Under an orthographic camera a body near the centre
       * of the disc has a normal pointing straight at the viewer, so its whole
       * height falls into `depth` and never reaches the screen — the stalk
       * collapses and the filled cap paints as a lens. And on the *lower* half
       * of the globe the normal points down the screen, so the mushroom grows
       * downward: an upside-down cap, hollow to the sky, with its own stalk
       * sticking out through the crown. That is the bronze crescent — not a
       * broken fill, a mushroom hanging by its head.
       *
       * A real mushroom does not grow along the surface normal, it grows
       * *up*. So does this one: the body always stands towards the top of the
       * screen, and the normal is used only to decide how much it leans, by
       * how far its footprint tips away from vertical. The globe still reads
       * as a sphere because the bodies near the limb splay outward, but none
       * of them is ever painted hanging.
       */
      const tip = project({ x: normal.x * (1 + size), y: normal.y * (1 + size), z: normal.z * (1 + size) }, 1);
      const bodyPixelsForFrame = radius * size;
      const outX = tip.sx - base.sx;
      const outLength = Math.hypot(outX, tip.sy - base.sy);
      /*
       * Take only the sideways component of the outward direction, and keep
       * "up" as up. That component is 0 at the centre of the disc and ±1 at
       * the limb; a third of it is the whole budget, because it is the *sine*
       * of the tilt. Passed through raw, a body at the limb leans 60° and is
       * a mushroom lying on its side — the splay has to be readable as a
       * sphere without any single body looking blown over. At 0.33 the limb
       * leans about 19°, which is roughly how far a real one leans anyway.
       */
      const swayLimit = 0.33;
      const sway = outLength > 1e-6 ? (outX / outLength) * swayLimit : 0;
      const upX = sway * bodyPixelsForFrame;
      const upY = -Math.sqrt(Math.max(0, 1 - sway * sway)) * bodyPixelsForFrame;
      // Width runs perpendicular to "up" on screen, so the cap is always seen
      // across its widest section — the outline of a solid of revolution.
      const rightX = -upY;
      const rightY = upX;
      const place = (point: Point) => ({
        sx: base.sx + upX * point.y + rightX * point.x,
        sy: base.sy + upY * point.y + rightY * point.x,
      });

      const body = growFruitingBody({ maturity: fruit.maturity, lean: fruit.lean, gillCount: fruit.gills });
      if (body.height <= 0) return;

      // Fades in with depth like the land dots, so a mushroom rotating into
      // view arrives rather than appears.
      const presence = Math.min(1, (base.depth - 0.16) / 0.22);
      const alpha = presence * (0.45 + base.depth * 0.55);
      /*
       * Stroke widths come off the body's own size, not off the globe's.
       * A mature body is `radius * FRUIT_SCALE` tall on screen, and the stipe
       * is about a seventh of that — the same proportion the closing mark
       * uses. Deriving it from the globe instead gave a one-pixel hairline
       * under a 20px cap, which reads as a parasol on a wire rather than as
       * something with a stem.
       */
      const bodyPixels = radius * FRUIT_SCALE;
      const stipeWidth = Math.max(1, bodyPixels * 0.15);
      const scale = Math.max(0.5, radius * 0.0022);

      const trace = (points: Point[]) => {
        context.beginPath();
        points.forEach((point, index) => {
          const { sx, sy } = place(point);
          if (index === 0) context.moveTo(sx, sy);
          else context.lineTo(sx, sy);
        });
      };

      // The stalk goes down first so the cap overprints where the two meet;
      // pale, because the bronze belongs to the cap. Butt-capped so its top
      // does not bulge a round end out through the crown.
      context.globalAlpha = alpha * 0.85;
      context.strokeStyle = "#e2d6c3";
      context.lineWidth = stipeWidth;
      context.lineCap = "butt";
      trace(body.stipe);
      context.stroke();
      context.lineCap = "round";

      /*
       * The cap: outline over the crown, then back along the underside, so the
       * path closes on a concave sweep rather than on a straight chord. That
       * one difference is what gives the rim an overhang to cast shadow under
       * — closed flat, the same fill is a bronze crescent.
       */
      context.globalAlpha = alpha;
      trace(body.cap);
      const undersideStart = body.underside[0];
      if (undersideStart) {
        for (const point of body.underside) {
          const { sx, sy } = place(point);
          context.lineTo(sx, sy);
        }
      }
      context.closePath();
      /*
       * Lit from up-screen, which is where the hero's own light comes from.
       * A flat fill at this size is a bronze sticker; two stops along the
       * body's own "up" axis are enough to round the dome, and because the
       * gradient is built on the billboard frame it stays consistent as the
       * globe turns.
       */
      const crown = place({ x: 0, y: body.apex.y + 0.55 });
      const rimLevel = place({ x: 0, y: body.apex.y - 0.25 });
      const shade = context.createLinearGradient(crown.sx, crown.sy, rimLevel.sx, rimLevel.sy);
      shade.addColorStop(0, "#e0a86a");
      shade.addColorStop(0.55, "#c98f52");
      shade.addColorStop(1, "#96633a");
      context.fillStyle = shade;
      context.fill();
      // A darker rim keeps the dome from flattening into a blob against the
      // land dots behind it.
      context.strokeStyle = "#8f5f31";
      context.lineWidth = Math.max(0.5, scale * 0.5);
      context.stroke();

      // Gills last and only on an open cap: at this scale they are two or
      // three pixels of texture under the rim, and they are the detail that
      // makes the shape read as a mushroom rather than as a pin. Drawn darker
      // than the fill — an underside is in shadow, and pale lines radiating
      // over a bronze dome are the ribs of a parasol.
      if (body.gills.length > 0) {
        context.globalAlpha = alpha * 0.42;
        context.strokeStyle = "#4a2d16";
        context.lineWidth = Math.max(0.4, bodyPixels * 0.035);
        // One path for the whole fan. `trace` opens a new path each call, so
        // stroking once after the loop drew only the last gill — the other six
        // were built and then discarded.
        context.beginPath();
        for (const [from, to] of body.gills) {
          const a = place(from);
          const b = place(to);
          context.moveTo(a.sx, a.sy);
          context.lineTo(b.sx, b.sy);
        }
        context.stroke();
      }
    };

    const render = () => {
      const radius = (Math.min(width, height) / 2) * 0.94;
      const centerX = width / 2;
      const centerY = height / 2;
      const cosRotation = Math.cos(rotation);
      const sinRotation = Math.sin(rotation);

      // Rotate around the pole, then tilt, then project orthographically.
      const project = (vector: Vector, scale: number) => {
        const x = vector.x * cosRotation + vector.z * sinRotation;
        const zr = vector.z * cosRotation - vector.x * sinRotation;
        const y = vector.y * cosTilt - zr * sinTilt;
        const z = vector.y * sinTilt + zr * cosTilt;
        return { sx: centerX + x * radius * scale, sy: centerY - y * radius * scale, depth: z };
      };

      context.clearRect(0, 0, width, height);

      context.beginPath();
      context.arc(centerX, centerY, radius, 0, Math.PI * 2);
      context.strokeStyle = "rgba(238,233,226,0.09)";
      context.lineWidth = 1;
      context.stroke();

      const dotRadius = Math.max(0.7, radius * 0.0038);
      for (const point of land) {
        const { sx, sy, depth } = project(point, 1);
        if (depth <= 0.02) continue;
        context.globalAlpha = 0.13 + depth * 0.5;
        context.beginPath();
        context.arc(sx, sy, dotRadius, 0, Math.PI * 2);
        context.fillStyle = "#e8e2d8";
        context.fill();
      }

      context.lineCap = "round";
      context.lineJoin = "round";
      const strandWidth = Math.max(0.75, radius * 0.0022);
      for (const filament of mycelium.filaments) {
        if (filament.progress <= 0 || filament.vitality <= 0) continue;

        /*
         * Sample the filament's own curve, not the great circle between its
         * ends. `filamentPoint` bows it sideways within the surface, which is
         * the difference between a hypha and a wire: a straight segment between
         * two dots is a graph edge, and every distributed-compute landing draws
         * those.
         */
        const path: Array<{ sx: number; sy: number; depth: number } | null> = [];
        let visibleDepth = 0;
        for (let step = 0; step <= ARC_SEGMENTS; step += 1) {
          const t = (step / ARC_SEGMENTS) * filament.progress;
          const projected = project(filamentPoint(filament, t), 1.006);
          if (projected.depth <= 0.02) {
            path.push(null);
            continue;
          }
          visibleDepth = Math.max(visibleDepth, projected.depth);
          path.push(projected);
        }
        if (visibleDepth <= 0) continue;

        const alpha = (0.18 + visibleDepth * 0.62) * filament.vitality;
        context.strokeStyle = filament.hub ? "rgba(198,146,90,0.95)" : "rgba(178,196,174,0.72)";

        /*
         * Stroked in bands rather than as one path, because a hypha is thickest
         * where it left its parent and finest at its growing tip. Canvas has no
         * variable-width stroke, and stroking each of the 26 segments on its
         * own costs more than the whole rest of the frame — four bands is where
         * the taper stops being visible as steps while the draw call count
         * stays in the hundreds rather than the thousands.
         */
        const bands = 4;
        const baseWidth = strandWidth * (filament.hub ? 1.5 : 1.05);
        for (let band = 0; band < bands; band += 1) {
          const from = Math.floor((band * ARC_SEGMENTS) / bands);
          const to = Math.floor(((band + 1) * ARC_SEGMENTS) / bands);
          // 1 at the root, ~0.3 at the tip.
          const thickness = 1 - 0.7 * ((band + 0.5) / bands);
          context.lineWidth = Math.max(0.35, baseWidth * thickness);
          context.globalAlpha = alpha * (0.72 + 0.28 * thickness);
          context.beginPath();
          let started = false;
          // Overlaps the next band by one sample so the bands butt rather than
          // leave a gap where the width changes.
          for (let step = from; step <= to; step += 1) {
            const point = path[step];
            if (!point) {
              started = false;
              continue;
            }
            if (started) context.lineTo(point.sx, point.sy);
            else context.moveTo(point.sx, point.sy);
            started = true;
          }
          context.stroke();
        }

        /*
         * Side hairs: short laterals that connect to nothing. They are what
         * makes a filament read as living tissue instead of as a route — the
         * network has texture along its length, not only at its junctions.
         * Drawn finer and fainter than the parent so they never compete with
         * the path the colony actually took.
         */
        if (filament.hairs.length > 0) {
          context.lineWidth = Math.max(0.3, strandWidth * 0.5);
          context.globalAlpha = alpha * 0.5;
          context.beginPath();
          let anyHair = false;
          for (const hair of filament.hairs) {
            if (hair.progress <= 0) continue;
            let started = false;
            for (let step = 0; step <= HAIR_SEGMENTS; step += 1) {
              const t = (step / HAIR_SEGMENTS) * hair.progress;
              const projected = project(hairPoint(filament, hair, t), 1.006);
              if (projected.depth <= 0.02) {
                started = false;
                continue;
              }
              if (started) context.lineTo(projected.sx, projected.sy);
              else context.moveTo(projected.sx, projected.sy);
              started = true;
              anyHair = true;
            }
          }
          if (anyHair) context.stroke();
        }

        /*
         * The advancing tip.
         *
         * A flat filled circle was the other half of the complaint, and it was
         * right: a hard-edged disc is a UI dot, and it is the one thing on this
         * globe with no organic edge at all. A hypha's tip is a swelling that
         * fades into its own halo, so this is a soft radial gradient with a
         * small bright core — the same treatment the page's own mycelium field
         * uses for its forks, so the two networks read as one organism.
         */
        const tip = path[path.length - 1] ?? null;
        if (tip && tip.depth > 0.02) {
          const growing = filament.progress < 1;
          const presence = (growing ? 0.9 : 0.55) * (0.35 + tip.depth * 0.65) * filament.vitality;
          // Growing tips swell, settled ones sit back down into the network.
          const size = (filament.hub ? dotRadius * 3.4 : dotRadius * 2.2) * (growing ? 1.15 : 1);
          const glow = context.createRadialGradient(tip.sx, tip.sy, 0, tip.sx, tip.sy, size);
          if (filament.hub) {
            glow.addColorStop(0, "rgba(247,226,196,0.95)");
            glow.addColorStop(0.32, "rgba(213,154,92,0.6)");
            glow.addColorStop(1, "rgba(213,154,92,0)");
          } else {
            glow.addColorStop(0, "rgba(233,238,228,0.85)");
            glow.addColorStop(0.34, "rgba(186,202,182,0.42)");
            glow.addColorStop(1, "rgba(186,202,182,0)");
          }
          context.globalAlpha = presence;
          context.beginPath();
          context.arc(tip.sx, tip.sy, size, 0, Math.PI * 2);
          context.fillStyle = glow;
          context.fill();

          // A small dense core inside the halo. Without it the tip is a smudge;
          // with it there is something the halo is the glow *of*.
          context.globalAlpha = presence * 0.9;
          context.beginPath();
          context.arc(tip.sx, tip.sy, Math.max(0.5, size * 0.26), 0, Math.PI * 2);
          context.fillStyle = filament.hub ? "#f0cfa4" : "#e2e8dc";
          context.fill();
        }
      }

      if (SHOW_GLOBE_FRUITING_BODIES) {
        for (const fruit of mycelium.fruits) drawFruit(fruit, project, radius);
      }
      context.globalAlpha = 1;
    };

    if (reduceMotion) {
      render();
      return () => observer.disconnect();
    }

    let frame = 0;
    let previous = performance.now();
    let inViewport = typeof IntersectionObserver === "undefined";
    const shouldAnimate = () => document.visibilityState === "visible" && inViewport;
    const loop = (now: number) => {
      frame = 0;
      if (!shouldAnimate()) return;
      // Clamped so a backgrounded tab does not resume with a growth jump.
      const delta = Math.min(0.05, Math.max(0, (now - previous) / 1000));
      previous = now;
      // Always the same direction, so the land never appears to rewind.
      rotation = (rotation + driftSpeed(rotation) * delta) % TAU;
      mycelium.advance(delta);
      render();
      frame = window.requestAnimationFrame(loop);
    };
    const updateAnimation = () => {
      if (!shouldAnimate()) {
        window.cancelAnimationFrame(frame);
        frame = 0;
        return;
      }
      previous = performance.now();
      if (!frame) frame = window.requestAnimationFrame(loop);
    };
    const visibilityObserver = typeof IntersectionObserver === "undefined" ? null : new IntersectionObserver((entries) => {
      inViewport = entries[0]?.isIntersecting ?? false;
      updateAnimation();
    });
    visibilityObserver?.observe(canvas);
    document.addEventListener("visibilitychange", updateAnimation);
    updateAnimation();

    return () => {
      window.cancelAnimationFrame(frame);
      visibilityObserver?.disconnect();
      document.removeEventListener("visibilitychange", updateAnimation);
      observer.disconnect();
    };
  }, []);

  return (
    <div className="rb-globe" aria-hidden="true">
      <canvas ref={canvasRef} />
    </div>
  );
}
