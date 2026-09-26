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
 * Phones do not draw the organism at all. Behind the globe it read as a
 * transparent ghost, and every earlier framing either overlapped the copy or
 * pushed it under the fold — so the mobile hero is the planet and the text,
 * and `wantsMushroom()` keeps the 190KB renderer off the phone entirely: no
 * prefetch, no WebGL context, no empty host.
 */

/** True when this visitor will actually draw the organism. */
export function wantsMushroom(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return true;
  return !window.matchMedia("(max-width: 700px)").matches;
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
    if (!host) return;

    const mobile = window.matchMedia("(max-width: 700px)");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let cancelled = false;
    let element: HTMLElement | null = null;
    let failsafe: number | undefined;
    let generation = 0;

    const stop = (resetShown: boolean) => {
      generation += 1;
      if (failsafe !== undefined) window.clearTimeout(failsafe);
      failsafe = undefined;
      element?.remove();
      element = null;
      if (resetShown) setShown(false);
    };
    const start = () => {
      if (element) return;
      const currentGeneration = ++generation;
      void loadMushroomStage().then(({ defineMushroomStage }) => {
        // Width changes and StrictMode unmounts can happen while the large
        // renderer is still downloading. Only the latest desktop request mounts.
        if (cancelled || mobile.matches || currentGeneration !== generation) return;
        defineMushroomStage();
        element = document.createElement("mushroom-stage");
        element.setAttribute("variant", "hero");
        element.setAttribute("accent", "#c9976a");
        element.setAttribute("spores", reduceMotion ? "off" : "on");
        element.setAttribute("motion", reduceMotion ? "off" : "full");
        element.setAttribute("scale", "1");
        element.style.cssText = "width:100%;height:100%;display:block;";
        const show = () => {
          if (currentGeneration !== generation) return;
          if (failsafe !== undefined) window.clearTimeout(failsafe);
          failsafe = undefined;
          setShown(true);
        };
        element.addEventListener("mushroom-ready", show, { once: true });
        // A failed WebGL boot must not leave the host transparent forever.
        failsafe = window.setTimeout(show, 4000);
        host.appendChild(element);
      }).catch(() => {
        // Decorative rendering can fail independently of the readable hero.
      });
    };
    const syncWidth = () => {
      if (wantsMushroom()) start();
      else stop(true);
    };
    mobile.addEventListener("change", syncWidth);
    syncWidth();

    return () => {
      cancelled = true;
      mobile.removeEventListener("change", syncWidth);
      stop(false);
    };
  }, []);

  return <div id="rb-hero-mushroom" className={`rb-hero-mushroom${shown ? " is-grown" : ""}`} ref={hostRef} aria-hidden="true" />;
}
