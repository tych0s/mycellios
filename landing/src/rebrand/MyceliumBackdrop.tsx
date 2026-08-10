import { useEffect, useRef } from "react";
import { growField, type Hypha } from "./mycelium-field";
import "./mycelium-backdrop.css";

/*
 * The mycelium that grows behind the page as you scroll.
 *
 * The brand is named after a fungal network, but until now that was only ever
 * stated — in the logo, the palette and the copy. This makes the page behave
 * like one: a colony that extends and branches while the reader descends, and
 * retracts if they go back up. It is a scrubber, not a triggered animation.
 *
 * Three constraints shaped every decision here.
 *
 * 1. It must not compete with the content. The whole point is that it is felt
 *    before it is inspected, so the ink stays translucent and filaments are pushed
 *    out of the centre column where the copy lives (see `mycelium-field.ts`),
 *    and the band fades out at both ends rather than stopping at a hard edge.
 *
 * 2. It must cross the existing sections. Every band paints its own opaque
 *    background, so a canvas behind `.rb-page` would simply be covered. This is
 *    a fixed, full-viewport canvas layered over the bands but under their copy.
 *    Its ink changes from bronze on paper to pale gold on forest green.
 *
 * 3. It must be free. The geometry is generated once per size and only walked
 *    per frame; work is scheduled in rAF rather than done in the scroll
 *    handler; and the renderer exits immediately while the band is off screen.
 */

/*
 * Alpha ceiling for a filament at full growth. Visible texture, never body ink.
 *
 * This is deliberately low. The failure mode of the previous pass was not
 * density but tone: a dark, desaturated ink multiplied into warm paper turns
 * grey, and sparse grey lines on beige read as cracks or cobwebs — the page
 * looked damaged rather than alive. The fix is the opposite of the intuitive
 * one: warmer, lighter ink drawn *more* densely. Many faint warm filaments
 * read as grain in the stock; few dark ones read as damage.
 *
 * The working range is narrow: 0.10 was nearly invisible on a bright screen,
 * while values much beyond 0.15 start asserting themselves over the paper.
 * This sits high enough to survive bright mobile displays while remaining a
 * substrate rather than a foreground illustration.
 */
const INK_ALPHA = 0.14;
const DARK_INK_MULTIPLIER = 1.35;
/*
 * Distance, in field pixels, over which a single filament is revealed. Long
 * enough that a segment always takes several scrolled frames to appear — the
 * reveal is what makes growth read as continuous rather than as strokes being
 * switched on — and short enough that the frontier still tracks the reader.
 */
const REVEAL_PX = 90;
const OCCLUDER_SELECTOR = [
  ".rb-pay-card",
  ".rb-pay-machines li",
  ".rb-rule",
  ".rb-local-term",
  ".rb-local-switch",
  ".rb-door",
  ".rb-board",
].join(",");
const SPAN_START = "how-it-works"; // the colony takes root where the story begins
const SPAN_END_SELECTOR = ".rb-sponsors"; // it dissolves after the dark install + closing sequence

export function MyceliumBackdrop() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const start = document.getElementById(SPAN_START);
    const end = document.querySelector<HTMLElement>(SPAN_END_SELECTOR);
    const darkStartElement = document.getElementById("install");
    if (!start || !end) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let field: Hypha[] = [];
    let fieldHeight = 1;
    let spanTop = 0;
    let spanHeight = 1;
    let darkStart = Number.POSITIVE_INFINITY;
    let darkEnd = Number.NEGATIVE_INFINITY;

    const measure = () => {
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      context.setTransform(ratio, 0, 0, ratio, 0, 0);

      // The colony spans document space, from the top of the story to the top
      // of the install band, so its growth is tied to the page rather than to
      // one viewport of scrolling.
      const scrollY = window.scrollY;
      spanTop = start.getBoundingClientRect().top + scrollY;
      spanHeight = Math.max(1, end.getBoundingClientRect().top + scrollY - spanTop);
      darkStart = darkStartElement
        ? darkStartElement.getBoundingClientRect().top + scrollY - spanTop
        : Number.POSITIVE_INFINITY;
      darkEnd = end.getBoundingClientRect().top + scrollY - spanTop;

      // Generated in viewport-width × span-height pixels, once per size.
      fieldHeight = spanHeight;
      field = growField(width, fieldHeight);
    };

    /*
     * `progress` is how far the reader has travelled through the span, measured
     * at the middle of the viewport: the colony's leading edge then sits near
     * the reader's eye rather than at the very bottom of the screen.
     */
    const render = () => {
      context.clearRect(0, 0, width, height);

      // Keep the fixed canvas dormant before the story and after the install
      // band. A passive scroll listener is still needed to notice when the
      // viewport enters this long document-space range.
      const viewportTop = window.scrollY;
      const viewportBottom = viewportTop + height;
      if (viewportBottom < spanTop || viewportTop > spanTop + spanHeight) return;

      const eye = window.scrollY + height * 0.5;
      const grown = Math.max(0, Math.min(1, (eye - spanTop) / spanHeight));
      if (grown <= 0) return;

      // Document-space y of the canvas' top edge, so the colony appears pinned
      // to the page while the canvas itself is fixed to the viewport.
      const offset = spanTop - window.scrollY;
      const frontier = grown * fieldHeight;

      context.lineCap = "round";
      for (const hypha of field) {
        if (hypha.y1 > frontier) break; // field is generated in growth order

        // The last stretch draws partially, so the tip advances smoothly
        // instead of snapping in one segment at a time.
        //
        // The reveal is spread over a fixed distance rather than over each
        // segment's own vertical span. Measuring against the span alone meant a
        // near-horizontal branch — which is most of them, since branches spread
        // sideways — had almost no span to travel and therefore snapped from
        // nothing to fully drawn between two frames. Dozens of those popping in
        // at once is exactly the jumping this needed to remove.
        const span = hypha.y2 - hypha.y1;
        const reveal = Math.max(span, REVEAL_PX);
        const t = Math.max(0, Math.min(1, (frontier - hypha.y1) / reveal));
        const y1 = hypha.y1 + offset;
        const x2 = hypha.x1 + (hypha.x2 - hypha.x1) * t;
        const y2 = y1 + span * t;

        // Cheap vertical cull: most of the colony is off screen at any moment.
        if ((y1 < -40 && y2 < -40) || (y1 > height + 40 && y2 > height + 40)) continue;

        // Older, thicker filaments carry the structure; branches fade back.
        // Ink in as well as extend: a filament arrives at full strength only
        // once it is fully drawn, so the growth frontier is a soft edge rather
        // than a line of hard new strokes advancing down the page.
        const depth = 1 - hypha.gen / 6;
        const documentMidpoint = hypha.y1 + span * t * 0.5;
        const overDarkBand = documentMidpoint >= darkStart && documentMidpoint < darkEnd;
        context.globalAlpha = INK_ALPHA * (overDarkBand ? DARK_INK_MULTIPLIER : 1) * (0.5 + depth * 0.5) * (0.35 + t * 0.65);
        context.lineWidth = 0.75 + depth * 0.7;
        // The ink adapts to the substrate: bronze deepens paper, while a pale
        // gold keeps the same organism legible across the forest-green bands.
        context.strokeStyle = overDarkBand
          ? (hypha.gen > 1 ? "#dfc29d" : "#e7b77e")
          : (hypha.gen > 1 ? "#c19a6e" : "#a8763f");
        context.beginPath();
        context.moveTo(hypha.x1, y1);
        const cx = hypha.x1 + (hypha.cx - hypha.x1) * t;
        const cy = y1 + (hypha.cy - hypha.y1) * t;
        context.quadraticCurveTo(cx, cy, x2, y2);
        context.stroke();

        // A fork that has been reached gets a faint bronze node: the only
        // colour in the whole backdrop, and the thing that makes it read as a
        // network rather than as cracks in the paper.
        // Fade the node in over the last stretch of its segment rather than
        // switching it on at `t === 1`. A dot that appears at full strength in
        // one frame is a step the eye catches, and dozens of them stepping in
        // as the reader scrolls was a large part of why growth looked jumpy.
        if (hypha.fork && t > 0.55) {
          context.globalAlpha = Math.min(overDarkBand ? 0.28 : 0.2, INK_ALPHA * (overDarkBand ? 1.9 : 1.5)) * ((t - 0.55) / 0.45);
          context.fillStyle = overDarkBand ? "#ecc18c" : "#ad7a48";
          context.beginPath();
          context.arc(x2, y2, 1.2, 0, Math.PI * 2);
          context.fill();
        }
      }

      // The organism belongs to the paper substrate. Opaque interface objects
      // sit above it, with a small breathing margin, so no filament can look
      // like a rendering defect drawn across a card.
      for (const element of document.querySelectorAll<HTMLElement>(OCCLUDER_SELECTOR)) {
        const rect = element.getBoundingClientRect();
        if (rect.bottom < 0 || rect.top > height) continue;
        context.clearRect(rect.left - 7, rect.top - 7, rect.width + 14, rect.height + 14);
      }
      context.globalAlpha = 1;
    };

    measure();

    if (reduceMotion) {
      // No scroll coupling at all: one still, fully-grown colony.
      const still = () => {
        measure();
        const saved = window.scrollY;
        // Place the synthetic eye exactly at the end of the field so render()
        // draws the complete static colony without scroll-linked movement.
        spanTop = saved + height * 0.5 - spanHeight;
        render();
      };
      still();
      window.addEventListener("resize", still);
      return () => window.removeEventListener("resize", still);
    }

    let frame = 0;
    const schedule = () => {
      if (!frame) {
        frame = window.requestAnimationFrame(() => {
          frame = 0;
          render();
        });
      }
    };
    const remeasure = () => {
      measure();
      schedule();
    };

    // Sections above the span change height on reveal, so the span is
    // re-measured rather than trusted from first paint.
    const layout = new ResizeObserver(remeasure);
    layout.observe(document.body);
    window.addEventListener("resize", remeasure);

    window.addEventListener("scroll", schedule, { passive: true });
    schedule();

    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", remeasure);
      layout.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  return <canvas className="rb-myco" ref={canvasRef} aria-hidden="true" />;
}
