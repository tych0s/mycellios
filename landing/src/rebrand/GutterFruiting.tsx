import { useEffect, useRef, useState } from "react";
import { growFruitingBody, type Point } from "./fruiting-body";
import { loadMushroomStage, wantsMushroom } from "./HeroMushroom";

/*
 * A young fruiting body coming up out of the exchange section's bottom edge.
 *
 * The mycelium already runs down this page's margins, so the question was never
 * whether to decorate the empty band but what the network is doing in it. This
 * answers it: the filament fruits. Same claim the page makes everywhere else —
 * the colony is one organism, and where it has finished work it produces
 * something.
 *
 * IT IS THE HERO'S RENDERER, NOT A DRAWING OF IT.
 *
 * That was the second attempt and the first was wrong. This started as a flat
 * SVG on the reasoning that a decoration beside a pricing table should not
 * compete with it — but "should not compete" means low contrast, not no volume,
 * and those are different things. The hero has already shown the visitor this
 * organism with real material on it; a flat one further down the same page does
 * not read as a quieter version of it, it reads as a worse one. So the body
 * below is `<mushroom-stage variant="gutter">`: same three.js renderer, same
 * lighting model, and it is kept subordinate by turning the emission down —
 * halo off, glow to a third, bloom threshold up — rather than by flattening it.
 *
 * And it is a *different* specimen, not the hero shrunk. Two identical
 * organisms on one page is wallpaper; a colony reads as alive because its
 * bodies are at different ages. The hero is a mature open cap, this is a young
 * closed bell — taller than it is wide, on a stem half the thickness, before
 * the cap has opened. The profile tables for both live in `VARIANTS` in the
 * renderer, and the hero's entry is the reference's own numbers verbatim,
 * pinned by a test so tuning this one cannot drift that one.
 *
 * The SVG below is not dead code and not a lesser copy: it is what the visitors
 * who never get WebGL actually see. `wantsMushroom()` excludes phones under
 * 700px (where three.js would be 133KB for a decoration behind the text) and
 * anyone with `prefers-reduced-motion` (where a breathing, rotating organism is
 * exactly what the setting exists to prevent). For them the flat silhouette is
 * the right answer rather than the fallback answer, so it keeps its own
 * reasoning:
 *
 * Drawn as outlines — cap arc, underside arc, a fan of gills, a hairline stalk
 * — it renders unmistakably as a *parasol*: an arc over a line with ribs
 * beneath it is what an umbrella is, and no tuning of the arc fixes it, because
 * what is missing is not a curve but the mass. The rim only reads as an
 * overhang once there is solid cap above it to overhang from. So the cap is a
 * closed filled silhouette, the stalk has real width, and the gills are
 * dropped.
 *
 * TRANSLUCENCY IS APPLIED TO THE GROUP, NEVER TO THE SHAPES.
 *
 * This is the one rule here that is not cosmetic. The stalk runs *behind* the
 * cap. Give each path its own alpha and the overlap composites twice, so the
 * stalk shows through the cap as a dark vertical stripe — the drawing stops
 * being an organism and becomes two transparent cut-outs laid on each other,
 * which is exactly what "se ve dibujada" means. The shapes are therefore opaque
 * against each other and the whole group carries a single `opacity`: the body
 * occludes itself like a solid object, and the group's alpha is what keeps it
 * quiet in the margin.
 *
 * The same reasoning gives the cap a shallow crown-to-rim ramp and the stalk a
 * taper: not modelling for its own sake, but the minimum that reads as a body
 * with a near side rather than a filled outline. It stays one hue.
 *
 * Young, too: at full maturity the cap flattens into a wide disc that wants to
 * be a logo, and the closing panel already provides the logo. At 0.68 the body
 * is still opening, which is what "growth" looks like in a section about work
 * being done — and it survives at 62px in a phone's corner, which the mature
 * silhouette does not.
 *
 * It renders from the shared generator rather than from a hand-drawn path so
 * that if the organism's proportions are ever re-tuned, this one changes with
 * the other three instead of silently becoming a different species.
 */

const MATURITY = 0.68;
const LEAN = -0.42;
/* A stalk has width. At a hairline this is a wire holding up a canopy — the
   same failure the closing mark records at 2.4 against a 106-wide cap. This is
   roughly a tenth of the cap's span, as on a real body. */
const STIPE_WIDTH = 14;
const SCALE = 78;

/*
 * The box is fitted to this specimen rather than copied from the closing mark,
 * whose pose is different: at maturity .68 with a -.42 lean the body spans
 * x -40..29 and y -70..0 at this scale, and the root runs 26 below the origin.
 * The bounds are those numbers plus half a stalk width of stroke room, so the
 * organism fills its box instead of floating inside one sized for another pose.
 * They are asserted against the generator in the tests rather than trusted,
 * because a box that no longer contains the specimen just clips the cap and
 * looks like a styling mistake.
 */
const PAD = STIPE_WIDTH / 2 + 4;
const VIEW_LEFT = -47;
const VIEW_WIDTH = 83;
const VIEW_TOP = -84;
const ROOT_DEPTH = 30;
const VIEW_HEIGHT = -VIEW_TOP + ROOT_DEPTH;

/*
 * Where the base of the stalk sits down the view box, 0..1 — the one number the
 * stylesheet needs from here. The wrapper's bottom edge is the section's rule,
 * and the svg is scaled by 1/this so the base lands exactly on it with the root
 * running underneath. Exported and asserted against the CSS rather than
 * duplicated there: the symptom of drift is an organism hovering above the line
 * or buried in it, which reads as a positioning bug rather than a stale number.
 */
export const ORIGIN_FRACTION = -VIEW_TOP / VIEW_HEIGHT; // 0.737

function path(points: Point[]): string {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"}${(point.x * SCALE).toFixed(2)} ${(-point.y * SCALE).toFixed(2)}`)
    .join(" ");
}

/*
 * The cap is one closed silhouette: the outline over the crown, then the
 * underside walked back to close it, then `Z`.
 *
 * This is the whole design of the mark, and it was arrived at by rasterising
 * the alternative rather than by preference. Drawn as separate open strokes —
 * a cap arc, an underside arc, a fan of gills, a stalk — the result is
 * unmistakably a parasol: an arc over a line with ribs under it is what an
 * umbrella *is*, and no amount of tuning the arc fixes it, because the missing
 * thing is not a curve but the mass. The rim only reads as an overhang when
 * there is solid cap above it for it to overhang from.
 *
 * So the underside is not a second line here, it is the bottom edge of the
 * body: `cap` out over the crown, `underside` back beneath it, closed. That
 * encloses a crescent with a hollow bitten out of its base — a dome with real
 * thickness, sitting over a cup — which is what the closing mark fills too.
 * The generator returns the two curves in exactly the order that concatenates.
 */
function capSilhouette(cap: Point[], underside: Point[]): string {
  return `${path(cap)} ${path(underside).replace(/^M/, "L")} Z`;
}

/*
 * The stalk as a closed outline rather than a fat stroke.
 *
 * A uniform stroke gives a stalk of constant width — a dowel. Real ones swell
 * towards the base where they meet the ground and narrow under the cap, and
 * that single change is most of what separates a drawn mushroom from a grown
 * one at this size. It also has to be a *fill* for the same reason the cap is:
 * a stroked path and a filled path with different alphas would composite
 * against each other, and the whole point here is that the body is one solid.
 *
 * Offsetting each sample along its own normal (central difference, so the
 * bend is followed rather than sheared) and walking back down the far side
 * closes it. The taper is eased so the swelling stays near the ground instead
 * of running the length of the stalk.
 */
function stipeOutline(centre: Point[], baseWidth: number, topWidth: number): string {
  const left: Point[] = [];
  const right: Point[] = [];
  for (let i = 0; i < centre.length; i += 1) {
    const here = centre[i];
    const previous = centre[Math.max(i - 1, 0)];
    const next = centre[Math.min(i + 1, centre.length - 1)];
    /* All three indices are clamped inside the array, so this is unreachable —
       it is here because the compiler cannot see that and a `!` would hide a
       real empty-centre bug behind a crash inside a render. */
    if (!here || !previous || !next) continue;
    const dx = next.x - previous.x;
    const dy = next.y - previous.y;
    const magnitude = Math.hypot(dx, dy) || 1;
    /* Normal in the generator's frame; `path()` flips y for both sides
       together, so the offset stays perpendicular after the flip. */
    const nx = -dy / magnitude;
    const ny = dx / magnitude;
    const t = i / (centre.length - 1);
    const half = (topWidth + (baseWidth - topWidth) * (1 - t) ** 1.7) / 2 / SCALE;
    left.push({ x: here.x - nx * half, y: here.y - ny * half });
    right.push({ x: here.x + nx * half, y: here.y + ny * half });
  }
  return `${path(left)} ${path([...right].reverse()).replace(/^M/, "L")} Z`;
}

/*
 * The flat specimen, for the visitors who never load the renderer.
 *
 * Everything above about mass, the closed silhouette and group alpha applies to
 * this and only this — the WebGL body has real geometry and none of those
 * problems.
 */
function GutterFruitingSvg() {
  /* Leaning away from the content column. `gillCount: 0` because gills are the
     other half of the parasol — lines radiating under an arc are ribs, and at
     this size they collapse into a smear anyway. The hollow the closed
     silhouette already encloses is what carries the underside here. */
  const body = growFruitingBody({ maturity: MATURITY, lean: LEAN, gillCount: 0 });
  const stipePath = stipeOutline([...body.stipe], STIPE_WIDTH * 1.15, STIPE_WIDTH * 0.72);
  const capPath = capSilhouette([...body.cap], [...body.underside]);
  const rimPath = path([...body.underside]);

  /* The stipe continues below the base into the mycelium's own lane, so the
     body is attached to the network rather than standing on nothing. Without
     it the mark floats: a mushroom needs to come out of a surface, and here
     the surface is the filament that is already drawn down this margin. */
  const rootY = ROOT_DEPTH - 4;
  const rootPath = `M0 0 C-3 ${rootY * 0.4} 5 ${rootY * 0.7} 1 ${rootY}`;

  return (
    /* `rb-reveal` only to borrow the page's single reveal observer — the CSS
       cancels its slide, because a body that translates in as a block reads as
       a card arriving rather than something growing out of the margin.
       `rb-decor` opts out of the stacking rule that makes every section child a
       positioned content layer; see mycelium-network.css. */
    <div className="rb-gutter-fruiting rb-decor rb-reveal" aria-hidden="true">
      <svg
        viewBox={`${VIEW_LEFT} ${VIEW_TOP} ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
        role="presentation"
        aria-hidden="true"
        focusable="false"
      >
        <defs>
          {/* Mobile keeps the cheap SVG, but it should still describe the same
              waxy organism as WebGL. An off-centre radial light gives the cap
              a crown and a shaded far side; the horizontal stipe ramp keeps
              the stem from reading as a strip of paper. */}
          <radialGradient id="rbGutterCap" cx="30%" cy="19%" r="86%">
            <stop offset="0" stopColor="#f0d7b8" />
            <stop offset="0.38" stopColor="#d4aa7b" />
            <stop offset="0.76" stopColor="#ad7543" />
            <stop offset="1" stopColor="#774823" />
          </radialGradient>
          <linearGradient id="rbGutterStipe" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stopColor="#a96f40" />
            <stop offset="0.38" stopColor="#ead0ae" />
            <stop offset="0.7" stopColor="#c49261" />
            <stop offset="1" stopColor="#8a552d" />
          </linearGradient>
          <filter id="rbGutterDepth" x="-35%" y="-25%" width="170%" height="165%">
            <feDropShadow dx="1.5" dy="2.5" stdDeviation="2.1" floodColor="#5b351b" floodOpacity="0.24" />
          </filter>
        </defs>
        {/* The body is one solid. Every shape inside this group is opaque
            against the others and the group alone carries the transparency —
            see the note at the top of this file. Per-shape alpha would let the
            stalk read straight through the cap as a dark stripe, which is the
            single thing that made the earlier version look drawn rather than
            grown. The root sits outside the group because it belongs to the
            mycelium below ground, and is fainter than the body. */}
        <path className="rb-gutter-root" d={rootPath} />
        <g className="rb-gutter-body" filter="url(#rbGutterDepth)">
          {/* Stalk before cap, so the cap's own mass covers where the stalk
              ends rather than the stalk standing proud of the crown. */}
          <path className="rb-gutter-stipe" d={stipePath} />
          <path className="rb-gutter-cap" d={capPath} />
          {/* The rim redrawn as a line over the fill. It is the one edge that
              has to survive when the whole mark is 60px in a phone's corner:
              the overhang is what separates a mushroom from a lollipop. */}
          <path className="rb-gutter-rim" d={rimPath} />
        </g>
      </svg>
    </div>
  );
}

/*
 * The WebGL specimen.
 *
 * Mounted on the same shared promise as the hero's, so the module is fetched
 * once for the page: by the time anyone has scrolled to the pricing rail the
 * chunk is long since resolved and awaiting it costs nothing. A second
 * instance is a second WebGL context and a few hundred triangles, not a second
 * download — which is the whole reason this can afford to be real geometry.
 *
 * It is mounted lazily on intersection rather than at page load. Not for the
 * bytes, which are shared, but because a WebGL context that renders every frame
 * behind three screens of scroll is a GPU cost with nothing on screen to show
 * for it. The element's own IntersectionObserver already skips its draw when
 * off screen; this avoids creating the context at all until the section is
 * within a viewport of arriving.
 */
function GutterFruitingStage() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    let cancelled = false;
    let element: HTMLElement | null = null;
    let failsafe = 0;

    const mount = () => {
      void loadMushroomStage().then(({ defineMushroomStage }) => {
        // The import resolves a microtask after unmount under StrictMode's
        // mount/unmount/remount, so the element must not be created if the
        // effect has already been cleaned up.
        if (cancelled) return;
        defineMushroomStage();
        element = document.createElement("mushroom-stage");
        /* `variant` chooses geometry and is read once at boot, so it has to be
           set before the element is connected. */
        element.setAttribute("variant", "gutter");
        element.setAttribute("accent", "#c9976a");
        /* No spores. They are the hero's signature and they drift upward across
           whatever is above them — here that is the pricing table. */
        element.setAttribute("spores", "off");
        /* Half amplitude. The hero is the thing you look at, so it may sway;
           this is in the corner of the eye, where full motion is a distraction
           rather than life. */
        element.setAttribute("motion", "calm");
        element.setAttribute("scale", "1");
        element.style.cssText = "width:100%;height:100%;display:block;";
        element.addEventListener("mushroom-ready", () => {
          setShown(true);
          window.clearTimeout(failsafe);
        }, { once: true });
        // A boot that fails (no WebGL, a lost context) would otherwise leave
        // the host at opacity 0 forever, so the fade is not allowed to depend
        // solely on an event that may never arrive.
        failsafe = window.setTimeout(() => setShown(true), 4000);
        host.appendChild(element);
      });
    };

    const observer = new IntersectionObserver((entries) => {
      if (!entries.some((entry) => entry.isIntersecting)) return;
      observer.disconnect();
      mount();
    }, { rootMargin: "100% 0px" });
    observer.observe(host);

    return () => {
      cancelled = true;
      observer.disconnect();
      window.clearTimeout(failsafe);
      element?.remove();
    };
  }, []);

  return (
    <div
      className={`rb-gutter-fruiting rb-gutter-stage rb-decor${shown ? " is-grown" : ""}`}
      ref={hostRef}
      aria-hidden="true"
    />
  );
}

/*
 * Which body this visitor gets.
 *
 * Decided once on mount rather than at module scope: `wantsMushroom()` reads
 * media queries, and evaluating it during render would make the component
 * depend on window in an environment that may not have one. The initial state
 * is the flat body, so the very first paint is always the one that needs
 * nothing, and desktop swaps to the renderer on the same tick the effect runs.
 */
export function GutterFruiting() {
  const [webgl, setWebgl] = useState(false);
  useEffect(() => setWebgl(wantsMushroom()), []);
  return webgl ? <GutterFruitingStage /> : <GutterFruitingSvg />;
}
