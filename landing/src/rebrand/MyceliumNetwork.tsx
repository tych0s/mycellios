import { useEffect, useRef } from "react";
import { defineMyceliumField } from "./vendor/mycelium-network.js";
import "./mycelium-network.css";

/*
 * The colony that falls out of the hero mushroom and runs the whole page.
 *
 * This replaces the earlier `MyceliumBackdrop`, which grew its own unrelated
 * network in a band between the story and the install section at under 10%
 * ink. Two networks on one page is one too many, and the old one had no
 * relationship to the organism it was supposedly the roots of.
 *
 * The renderer is `vendor/mycelium-network.js` — no dependencies, plain
 * Canvas2D. What earns its keep is `origin-from`: the field looks up the
 * hero mushroom host, takes its box, and anchors the primary roots at the
 * base of the stem. The mycelium is then literally growing out of the
 * mushroom rather than being drawn near it, which is the entire metaphor and
 * the thing you cannot get by placing two decorations on the same page.
 *
 * It is scroll-driven and monotonic: the network extends as you descend and
 * does not retract when you go back up. That is deliberate and it is how a
 * colony behaves — it is a record of where the reader has been, not an
 * animation tied to a scrollbar position.
 */

export function MyceliumNetwork() {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    defineMyceliumField();
    const field = document.createElement("mycelium-field");
    field.setAttribute("accent", "#ad7a48");
    field.setAttribute("density", "0.58");
    /*
     * The renderer's own default is additive light on a black page. This
     * landing alternates dark and ivory bands, and adding light to paper is a
     * no-op, so `dim` switches it to normal compositing with heavier ink —
     * bronze that reads as tint on the light bands and still as a filament on
     * the dark ones.
     */
    field.setAttribute("dim", "");
    /*
     * Anchor to the stable host rather than the dynamically imported custom
     * element. The root geometry is therefore correct on the first layout,
     * even while the WebGL fruiting body is still compiling its first frame.
     */
    field.setAttribute("origin-from", "#rb-hero-mushroom");
    field.setAttribute(
      "motion",
      window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "off" : "full",
    );
    host.appendChild(field);

    return () => field.remove();
  }, []);

  return <div className="rb-myco-net" ref={hostRef} aria-hidden="true" />;
}
