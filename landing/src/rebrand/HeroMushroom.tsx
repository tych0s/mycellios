import { useEffect, useRef, useState } from "react";

/*
 * The hero's standing fruiting body — the brand's organism at full size,
 * behind the headline.
 *
 * The renderer is a custom element in `vendor/mushroom-stage.js`: three.js,
 * revolved profiles, translucent physical material, instanced gills, falling
 * spores and a hand-rolled bloom chain. This file is only the mount point and
 * the loading policy, and it deliberately stays that thin.
 *
 * three.js is ~190KB gzipped, which is more than the rest of this landing put
 * together. So it is imported *dynamically*: the headline, the lede and the
 * prompt all render and are readable before a single byte of the renderer is
 * needed.
 *
 * The cost of that split used to be a visible one-to-two second gap: the chunk
 * could not begin downloading until the main bundle had been fetched, parsed
 * and mounted, so two large downloads happened strictly one after the other and
 * the organism arrived long after the text it belongs with. `startMushroom()`
 * in main.tsx now kicks the same import off as the page boots, so the two
 * downloads overlap; by the time this effect runs the module is usually already
 * in flight or resolved, and awaiting the shared promise costs nothing.
 *
 * What remains after the download is shader compilation, which cannot be
 * removed — so it is covered instead of hidden: the host fades in over 600ms on
 * the element's `mushroom-ready` event, which fires on its first *drawn frame*.
 * An organism that materialises out of the dark reads as intended; the same
 * organism appearing between two frames reads as a bug.
 *
 * Two cases skip the download entirely rather than paying for it and then
 * doing nothing with it:
 *   - `prefers-reduced-motion`, where a rotating, breathing, spore-shedding
 *     organism is exactly what the setting exists to prevent;
 *   - viewports at or below 700px, where the copy occupies the full width and
 *     the mushroom would sit behind the text rather than beside it, so a mobile
 *     visitor would be charged three quarters of the page weight for a
 *     decoration they cannot see.
 *
 * That 700 is the same number as the `max-width:700px` block in rebrand.css
 * that hides `.rb-hero-mushroom`, and the same number `wantsMushroom()` uses to
 * decide whether to prefetch at all. All three must agree: a gap between them
 * is either a band of widths where the host paints its shade plate across the
 * headline with no organism inside it, or a phone paying for a chunk it will
 * never render.
 */

/** True when this visitor will actually draw the organism. */
export function wantsMushroom(): boolean {
  return (
    !window.matchMedia("(prefers-reduced-motion: reduce)").matches &&
    !window.matchMedia("(max-width: 700px)").matches
  );
}

let stagePromise: Promise<typeof import("./vendor/mushroom-stage.js")> | null = null;

/**
 * Begins loading the renderer. Called once from the page bootstrap so the
 * download overlaps the main bundle instead of queueing behind it, and again
 * from the component — the promise is shared, so the second call is free.
 */
export function loadMushroomStage(): Promise<typeof import("./vendor/mushroom-stage.js")> {
  stagePromise ??= import("./vendor/mushroom-stage.js");
  return stagePromise;
}

export function HeroMushroom() {
  const hostRef = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(false);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !wantsMushroom()) return;

    let cancelled = false;
    let element: HTMLElement | null = null;

    void loadMushroomStage().then(({ defineMushroomStage }) => {
      // The import resolves a microtask after unmount in StrictMode's
      // mount/unmount/remount, so the element must not be created if the
      // effect has already been cleaned up.
      if (cancelled) return;
      defineMushroomStage();
      element = document.createElement("mushroom-stage");
      element.setAttribute("accent", "#c9976a");
      element.setAttribute("spores", "on");
      element.setAttribute("motion", "full");
      element.setAttribute("scale", "1");
      element.style.cssText = "width:100%;height:100%;display:block;";
      element.addEventListener("mushroom-ready", () => setShown(true), { once: true });
      // A boot that fails (no WebGL, a lost context) would otherwise leave the
      // host at opacity 0 forever, so the fade is not allowed to depend solely
      // on an event that may never arrive.
      const failsafe = window.setTimeout(() => setShown(true), 4000);
      element.addEventListener("mushroom-ready", () => window.clearTimeout(failsafe), { once: true });
      host.appendChild(element);
    });

    return () => {
      cancelled = true;
      element?.remove();
    };
  }, []);

  return <div className={`rb-hero-mushroom${shown ? " is-grown" : ""}`} ref={hostRef} aria-hidden="true" />;
}
