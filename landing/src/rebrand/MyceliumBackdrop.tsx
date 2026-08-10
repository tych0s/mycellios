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
 *    and not looked at, so the ink stays under 10% alpha, filaments are pushed
 *    out of the centre column where the copy lives (see `mycelium-field.ts`),
 *    and the band fades out at both ends rather than stopping at a hard edge.
 *
 * 2. It cannot touch the existing sections. Every band on this page paints its
 *    own opaque background, so a canvas behind `.rb-page` would simply be
 *    covered. Instead this is a fixed, full-viewport canvas layered *over* the
 *    light bands but under the text, painting ink so faint it reads as tint in
 *    the paper. That is why it must end before the forest-green install
 *    section: light ink on a dark band would invert into a smudge.
 *
 * 3. It must be free. The geometry is generated once per size and only walked
 *    per frame; work is scheduled in rAF rather than done in the scroll
 *    handler; and the loop does not run at all while the band is off screen.
 */

/** Alpha ceiling for a filament at full growth. Tint, not line art. */
const INK_ALPHA = 0.093;
const SPAN_START = "how-it-works"; // the colony takes root where the story begins
const SPAN_END = "install"; // …and is gone before the dark download band

export function MyceliumBackdrop() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const start = document.getElementById(SPAN_START);
    const end = document.getElementById(SPAN_END);
    if (!start || !end) return;

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    let field: Hypha[] = [];
    let fieldHeight = 1;
    let spanTop = 0;
    let spanHeight = 1;

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
        const span = hypha.y2 - hypha.y1;
        const t = span <= 0 ? 1 : Math.max(0, Math.min(1, (frontier - hypha.y1) / span));
        const y1 = hypha.y1 + offset;
        const x2 = hypha.x1 + (hypha.x2 - hypha.x1) * t;
        const y2 = y1 + span * t;

        // Cheap vertical cull: most of the colony is off screen at any moment.
        if ((y1 < -40 && y2 < -40) || (y1 > height + 40 && y2 > height + 40)) continue;

        // Older, thicker filaments carry the structure; branches fade back.
        const depth = 1 - hypha.gen / 5;
        context.globalAlpha = INK_ALPHA * (0.45 + depth * 0.55);
        context.lineWidth = 0.75 + depth * 0.7;
        context.strokeStyle = "#1e2a22";
        context.beginPath();
        context.moveTo(hypha.x1, y1);
        context.lineTo(x2, y2);
        context.stroke();

        // A fork that has been reached gets a faint bronze node: the only
        // colour in the whole backdrop, and the thing that makes it read as a
        // network rather than as cracks in the paper.
        if (hypha.fork && t >= 1) {
          context.globalAlpha = INK_ALPHA * 2.1;
          context.fillStyle = "#ad7a48";
          context.beginPath();
          context.arc(x2, y2, 1.6, 0, Math.PI * 2);
          context.fill();
        }
      }
      context.globalAlpha = 1;
    };

    measure();

    if (reduceMotion) {
      // No scroll coupling at all: one still, fully-grown colony.
      const still = () => {
        measure();
        const saved = window.scrollY;
        spanTop = saved - spanHeight * 0.5; // force progress to 1
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

    // The loop is only wired up while the span is actually on screen.
    let listening = false;
    const listen = (on: boolean) => {
      if (on === listening) return;
      listening = on;
      if (on) {
        window.addEventListener("scroll", schedule, { passive: true });
        schedule();
      } else {
        window.removeEventListener("scroll", schedule);
        context.clearRect(0, 0, width, height);
      }
    };

    const visibility = new IntersectionObserver(
      (entries) => entries.forEach((entry) => listen(entry.isIntersecting)),
      { rootMargin: "20% 0px" },
    );
    // Observing a document-space proxy would need a wrapper element; the two
    // anchors bracket the span, so watching both covers entering from either
    // end and the long middle where neither is on screen.
    visibility.observe(start);
    visibility.observe(end);

    // Sections above the span change height on reveal, so the span is
    // re-measured rather than trusted from first paint.
    const layout = new ResizeObserver(remeasure);
    layout.observe(document.body);
    window.addEventListener("resize", remeasure);

    // The span usually starts on screen at load in the middle of the page.
    listen(true);
    schedule();

    return () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", remeasure);
      visibility.disconnect();
      layout.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  return <canvas className="rb-myco" ref={canvasRef} aria-hidden="true" />;
}
