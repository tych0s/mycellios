import { useEffect, useRef, useState } from "react";
import { loadMushroomStage, wantsMushroom } from "./HeroMushroom";

/*
 * Three fruiting bodies in the questions column.
 *
 * ONE ELEMENT, NOT THREE.
 *
 * The obvious build — three <mushroom-stage> elements side by side — is the
 * wrong one twice over. Technically, it would put five live WebGL contexts on
 * one page next to the hero and the gutter, and browsers keep somewhere between
 * eight and sixteen before they start dropping the oldest; a page that kills
 * its own hero once you scroll far enough is not a page that has decoration, it
 * is a page that has a bug. And pictorially, three canvases are three separate
 * pictures of a mushroom — each with its own camera and its own light, none of
 * them standing anywhere in particular relative to the others.
 *
 * So the renderer grew a `colony`: three bodies in one scene, sharing a camera,
 * a key light and a ground plane. That sharing is the entire reason the group
 * reads as a colony rather than as a row of ornaments, and it costs two extra
 * draw calls rather than two extra contexts.
 *
 * WHY THREE AGES AND NOT THREE MUSHROOMS.
 *
 * The same reason the gutter specimen is not the hero shrunk. Identical bodies
 * are wallpaper; what makes a patch of ground look alive is that the things
 * growing out of it are at different points in the same life. So one is old with
 * its cap turned up at the brim, one is mid-tear, and one is a button that has
 * barely cleared the surface — and the youngest is small and in front, which is
 * where the youngest one actually is.
 *
 * WHY IT IS QUIETER THAN THE GUTTER'S, WHICH WAS ALREADY QUIET.
 *
 * This one stands beside six rows of prose that a visitor is *reading*, which is
 * a harder neighbour than a pricing table you scan. The halo is off and the
 * glow is a third, because low contrast is what keeps a thing subordinate — not
 * taking the volume out of it, which is the mistake this whole line of work
 * started by making.
 *
 * NO FALLBACK BODY.
 *
 * Unlike the gutter, there is no SVG underneath. The gutter's flat specimen
 * earns its place because it sits in a margin the layout has already reserved,
 * so something has to be there. This is a block in the middle of a column: a
 * visitor on a phone, or one who asked for reduced motion, gets a column with
 * a heading and answers in it, which is a complete design rather than a
 * degraded one. Drawing three flat mushrooms for them would be inventing work
 * for the visitor least able to afford it.
 */

export function QuestionsColony() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !wantsMushroom()) return;

    const section = host.closest<HTMLElement>(".rb-quest");
    const heading = section?.querySelector<HTMLElement>(".rb-quest-head");
    let motionFrame = 0;

    /* THEY STAND STILL.
     *
     * An earlier version slid the colony left as the sticky heading released.
     * The travel was the whole problem: it was motion attached to nothing the
     * reader is doing — they are reading answers, not watching the margin — and
     * every pixel of it was a chance to land on the column it had to avoid. A
     * planted thing that stays planted reads as ground; a planted thing that
     * creeps reads as a bug, even when the arithmetic is right.
     *
     * So there is no progress, no release ramp and no scroll listener. The only
     * horizontal number left is the one that puts the bodies under the heading
     * column, and it is recomputed on layout changes alone. */
    const positionColony = () => {
      motionFrame = 0;
      if (!section || !heading) return;
      // MEASURE IN THE BOX THAT `left` IS RESOLVED AGAINST.
      //
      // The colony is absolutely positioned, so `left` is resolved against its
      // offsetParent — the centred shell, not the full-bleed section. Measuring
      // the heading in section space and writing the result as `left` is off by
      // exactly the shell's own inset. That inset is 0 while the shell still
      // fills the viewport, which is why this looked correct at 1440px and
      // shipped; at 1720px it is 212px, and 212px to the right is on top of the
      // answers column.
      const originBox = (host.offsetParent ?? section).getBoundingClientRect();
      const headingBox = heading.getBoundingClientRect();
      // The renderer is centred inside a box wider than the copy column. Centre
      // the *visible colony* under that column instead of aligning the canvas'
      // left edge, which incorrectly placed the bodies beneath the FAQs.
      const startCenter = headingBox.left - originBox.left + headingBox.width / 2;
      const startLeft = startCenter - host.offsetWidth / 2;
      host.style.setProperty("--rb-quest-colony-left", `${startLeft}px`);
    };

    const schedulePosition = () => {
      if (!motionFrame) motionFrame = window.requestAnimationFrame(positionColony);
    };
    /* Layout only. Scroll cannot change where this belongs, so scroll is not
       listened to — which also means no per-frame work while the section is
       being read. */
    const resizeObserver = new ResizeObserver(schedulePosition);
    resizeObserver.observe(section ?? host);
    window.addEventListener("resize", schedulePosition);
    positionColony();

    let cancelled = false;
    let element: HTMLElement | null = null;
    let failsafe = 0;

    const mount = () => {
      void loadMushroomStage().then(({ defineMushroomStage }) => {
        // The import resolves a microtask after unmount under StrictMode's
        // mount/unmount/remount, so the element must not be created if the
        // effect has already been torn down.
        if (cancelled) return;
        defineMushroomStage();
        element = document.createElement("mushroom-stage");
        element.setAttribute("variant", "colony");
        element.setAttribute("accent", "#c9976a");
        // Spores would drift up across the answers in the next column.
        element.setAttribute("spores", "off");
        element.setAttribute("motion", "calm");
        element.setAttribute("scale", "1");
        element.style.cssText = "width:100%;height:100%;display:block;";
        // Fade on the first *drawn* frame rather than on mount, so the group
        // does not pop in as a blank box that fills a beat later.
        element.addEventListener(
          "mushroom-ready",
          () => {
            setShown(true);
            window.clearTimeout(failsafe);
          },
          { once: true },
        );
        // If the renderer never reports a frame — a lost context, a driver that
        // refuses the shaders — the block stays at opacity 0 forever. Reveal it
        // anyway: an empty space is a better failure than a permanent hole.
        failsafe = window.setTimeout(() => setShown(true), 4000);
        host.appendChild(element);
      });
    };

    /* Three bodies is more geometry than the gutter's one, and this section is
       most of a page down, so nothing is built until the column is nearly in
       view. `rootMargin` gives it a viewport of warning, which is enough for
       the boot to finish before it is looked at. */
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        mount();
      },
      { rootMargin: "100% 0px" },
    );
    observer.observe(host);

    return () => {
      cancelled = true;
      window.cancelAnimationFrame(motionFrame);
      resizeObserver.disconnect();
      window.removeEventListener("resize", schedulePosition);
      observer.disconnect();
      window.clearTimeout(failsafe);
      element?.remove();
    };
  }, []);

  return (
    <div
      className={`rb-quest-colony${shown ? " is-grown" : ""}`}
      ref={hostRef}
      aria-hidden="true"
    />
  );
}
