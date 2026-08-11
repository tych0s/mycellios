import { growFruitingBody, type Point } from "./fruiting-body";

/*
 * The fruiting body as a still mark, for the closing panel.
 *
 * Same geometry as the hero globe and the page backdrop, in a third medium.
 * That is deliberate: three sizes of the same organism, drawn by three
 * different renderers from one generator, is what makes this read as a visual
 * system rather than as three mushrooms someone added to a page.
 *
 * SVG rather than canvas here because this one is static and it is the last
 * thing on the page. It renders into the server-side markup, so it is in the
 * HTML a crawler sees and it is painted on first frame with no script — the
 * closing panel is where a reader lands after the whole scroll, and a mark
 * that pops in after hydration would be the one they notice loading.
 *
 * It is aria-hidden. The panel below it already says "Many machines. One
 * model." — the mark restates that visually and announcing it twice to a
 * screen reader would be noise.
 */

/* Local units run 0…1 up and, once the cap is open, ±0.68 across — a mature
   body is wider than it is tall. The box is sized to that: 120 across for a
   106-wide cap, and the y window runs from just above the crown down to the
   base of the stipe rather than being a square, so the mark is not padded with
   empty space it would then be scaled down to fit.

   The margins are slack, not tight. The geometry reaches y = -93.9 at this
   scale, and a window starting at exactly -94 clipped the crown flat the
   moment the cap's proportions changed — the stroke on the outline alone is
   2.6 units wide and sits *outside* the path. A few units of air at the top
   costs nothing and keeps the mark whole if the dome is ever re-tuned. */
const VIEW_WIDTH = 124;
const VIEW_TOP = -100;
const VIEW_HEIGHT = 108;
const SCALE = 78;

function path(points: Point[]): string {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${(point.x * SCALE).toFixed(2)} ${(-point.y * SCALE).toFixed(2)}`)
    .join(" ");
}

export function FruitingMark() {
  // Fully open, leaning slightly, drawn once at module scale — nothing here
  // depends on the browser, which is what lets it server-render.
  const body = growFruitingBody({ maturity: 1, lean: -0.22, gillCount: 9 });

  return (
    <svg
      className="rb-fruit-mark"
      viewBox={`${-VIEW_WIDTH / 2} ${VIEW_TOP} ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
      role="presentation"
      aria-hidden="true"
      focusable="false"
    >
      {/*
        Order is the order it grows in, and it is also the order that makes the
        shape legible: stalk first so the cap overprints the join, then the cap
        as a *filled* dome, then the gills on top of that fill.

        The fill is the whole difference between a mushroom and an umbrella. An
        open arc stroked on its own is a rainbow over a stick no matter how the
        curve is tuned — and the gills under it then read as the ribs, which is
        the exact wrong reading. Closing the cap into a solid gives the stalk
        something to disappear behind, which is what a cap does.
      */}
      <path className="rb-fruit-stipe" d={path(body.stipe)} />
      {/* Cap outline and underside are one path: closing over the crown and
          back along the hollow gives the rim an overhang, where closing on a
          straight chord would fill a lens. */}
      <path className="rb-fruit-cap" d={`${path(body.cap)} ${path(body.underside).replace(/^M/, "L")} Z`} />
      <g className="rb-fruit-gills">
        {body.gills.map(([from, to], index) => (
          <path key={index} d={path([from, to])} />
        ))}
      </g>
    </svg>
  );
}
