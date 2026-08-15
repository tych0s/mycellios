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
 * Every viewport draws the same organism. Mobile changes only its rendering
 * budget and motion, never its geometry or material identity.
 */

/** True when this visitor will actually draw the organism. */
export function wantsMushroom(): boolean {
  return true;
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

    const mobile = window.matchMedia("(max-width: 700px)").matches;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let cancelled = false;
    let element: HTMLElement | null = null;

    void loadMushroomStage().then(({ defineMushroomStage }) => {
      // The import resolves a microtask after unmount in StrictMode's
      // mount/unmount/remount, so the element must not be created if the
      // effect has already been cleaned up.
      if (cancelled) return;
      defineMushroomStage();
      element = document.createElement("mushroom-stage");
      element.setAttribute("variant", "hero");
      if (mobile) element.setAttribute("quality", "mobile");
      element.setAttribute("accent", "#c9976a");
      element.setAttribute("spores", mobile || reduceMotion ? "off" : "on");
      element.setAttribute("motion", reduceMotion ? "off" : mobile ? "calm" : "full");
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

  return <div id="rb-hero-mushroom" className={`rb-hero-mushroom${shown ? " is-grown" : ""}`} ref={hostRef} aria-hidden="true" />;
}
