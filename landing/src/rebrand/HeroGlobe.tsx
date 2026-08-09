import { useEffect, useRef } from "react";
import { buildLandVectors, GROWTH_SECONDS, Mycelium, slerp, type Vector } from "./mycelium-growth";

/*
 * Dot globe for the hero: the world drawn as land points, connected by a
 * mycelium that grows across continents instead of a fixed decorative mesh.
 *
 * Two motions run at once and both are deliberately slow:
 *  - the globe turns steadily eastward, so land always travels to the right;
 *  - filaments extend hypha by hypha from a few seeds, branching into nearby
 *    land, so the network is visibly larger a minute after the page loads.
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
const ARC_SEGMENTS = 18;
const TAU = Math.PI * 2;

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

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    let rotation = DRIFT_START;
    const cosTilt = Math.cos(TILT);
    const sinTilt = Math.sin(TILT);

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
      for (const filament of mycelium.filaments) {
        if (filament.progress <= 0) continue;
        let started = false;
        let visibleDepth = 0;
        context.beginPath();
        for (let step = 0; step <= ARC_SEGMENTS; step += 1) {
          const t = (step / ARC_SEGMENTS) * filament.progress;
          const { sx, sy, depth } = project(slerp(filament.from, filament.to, t), 1.006);
          if (depth <= 0.02) {
            started = false;
            continue;
          }
          visibleDepth = Math.max(visibleDepth, depth);
          if (started) context.lineTo(sx, sy);
          else context.moveTo(sx, sy);
          started = true;
        }
        if (visibleDepth <= 0) continue;
        context.globalAlpha = 0.18 + visibleDepth * 0.62;
        context.strokeStyle = filament.hub ? "rgba(198,146,90,0.95)" : "rgba(180,196,176,0.7)";
        context.lineWidth = filament.hub ? 1.15 : 0.85;
        context.stroke();

        // The advancing tip reads as the living edge of the colony.
        const tip = project(slerp(filament.from, filament.to, filament.progress), 1.006);
        if (tip.depth > 0.02) {
          const growing = filament.progress < 1;
          context.globalAlpha = (growing ? 0.85 : 0.5) * (0.35 + tip.depth * 0.65);
          context.beginPath();
          context.arc(tip.sx, tip.sy, filament.hub ? dotRadius * 2.3 : dotRadius * 1.5, 0, Math.PI * 2);
          context.fillStyle = filament.hub ? "#d59a5c" : "#cfd8c9";
          context.fill();
        }
      }
      context.globalAlpha = 1;
    };

    if (reduceMotion) {
      render();
      return () => observer.disconnect();
    }

    let frame = 0;
    let previous = performance.now();
    const loop = (now: number) => {
      // Clamped so a backgrounded tab does not resume with a growth jump.
      const delta = Math.min(0.05, Math.max(0, (now - previous) / 1000));
      previous = now;
      // Always the same direction, so the land never appears to rewind.
      rotation = (rotation + driftSpeed(rotation) * delta) % TAU;
      mycelium.advance(delta);
      render();
      frame = window.requestAnimationFrame(loop);
    };
    frame = window.requestAnimationFrame(loop);

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, []);

  return (
    <div className="rb-globe" aria-hidden="true">
      <canvas ref={canvasRef} />
    </div>
  );
}
