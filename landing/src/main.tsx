import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Panel } from "./Panel";
import { NetworkPage } from "./NetworkPage";
import { RebrandLanding } from "./rebrand/RebrandLanding";
import { CreateStudio } from "./rebrand/CreateStudio";
import { EarnStudio } from "./rebrand/EarnStudio";
import { SporeStudio } from "./rebrand/SporeStudio";
import { loadMushroomStage, wantsMushroom } from "./rebrand/HeroMushroom";
import { applySeoMetadata } from "./seo";
import { resolveLandingSurface } from "./routing";

const root = document.getElementById("root");
if (!root) throw new Error("Missing landing root element");

// Phase 2 (merged landing): `/rebrand` no longer serves a separate concept — the
// rebrand design is the landing. Redirect to `/` before applying SEO metadata so
// the canonical tags describe the merged page.
const isRebrandRoute = window.location.pathname === "/rebrand" || window.location.pathname === "/rebrand/";
if (isRebrandRoute) window.history.replaceState({}, "", "/");
applySeoMetadata();

const path = window.location.pathname;
const surface = resolveLandingSurface(path, window.location.search);

const page = path === "/create" ? <CreateStudio />
  : path === "/earn" ? <EarnStudio />
  : path.startsWith("/spore") ? <SporeStudio />
  : surface === "network" ? <NetworkPage />
  : surface === "panel" ? <Panel />
  : <RebrandLanding />;

/*
 * Start the hero renderer's download here rather than letting the component ask
 * for it after mount.
 *
 * The mushroom chunk is the largest asset on the page, and a dynamic import
 * inside an effect cannot begin until this bundle has been fetched, parsed and
 * rendered — so the two biggest downloads ran strictly back to back and the
 * organism landed a second or two after the text it belongs with. Requesting it
 * now overlaps them: same chunk, same lazy boundary, no bytes added to this
 * bundle, but the network is busy with it while React is still mounting.
 *
 * The condition is derived from the element actually being rendered, not from a
 * second copy of the routing rules — a prefetch guarded by its own paraphrase of
 * the router is a phone downloading 190KB it will never draw, one refactor from
 * now. `wantsMushroom()` then applies the same reduced-motion and width tests
 * the component uses.
 */
if (page.type === RebrandLanding && wantsMushroom()) {
  void loadMushroomStage().catch(() => {
    /* The component awaits the same shared promise and reports there. */
  });
}

createRoot(root).render(<StrictMode>{page}</StrictMode>);

const loader = document.getElementById("mycellios-loader");
if (loader) {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    loader.dataset.state = "leaving";
    loader.addEventListener("transitionend", () => loader.remove(), { once: true });
    window.setTimeout(() => loader.remove(), 400);
  }));
}
